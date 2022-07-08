import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from pathlib import Path

from utils import experiment_manager


def create_network(cfg):
    if cfg.MODEL.TYPE == 'unet':
        model = UNet(cfg)
    elif cfg.MODEL.TYPE == 'dualstreamunet':
        model = DualStreamUNet(cfg)
    elif cfg.MODEL.TYPE == 'reconstructionnet_v1':
        model = ReconstructionNetV1(cfg)
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    return nn.DataParallel(model)


def save_checkpoint(network, optimizer, epoch, step, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch: float, cfg: experiment_manager.CfgNode, device: str, net_file: Path = None):
    net = create_network(cfg)
    net.to(device)

    if net_file is None:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
        checkpoint = torch.load(save_file, map_location=device)
    else:
        checkpoint = torch.load(net_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY
        if cfg.DATALOADER.INPUT_MODE == 's1':
            n_channels = len(cfg.DATALOADER.S1_BANDS)
        elif cfg.DATALOADER.INPUT_MODE == 's2':
            n_channels = len(cfg.DATALOADER.S2_BANDS)
        else:
            n_channels = len(cfg.DATALOADER.S1_BANDS) + len(cfg.DATALOADER.S2_BANDS)

        self.inc = InConv(n_channels, topology[0], DoubleConv)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.outc = OutConv(topology[0], 1)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor) -> tuple:
        if self.cfg.DATALOADER.INPUT_MODE == 's1':
            x = x_s1
        elif self.cfg.DATALOADER.INPUT_MODE == 's2':
            x = x_s2
        else:
            x = torch.cat((x_s1, x_s2), dim=1)
        features = self.inc(x)
        features = self.encoder(features)
        features = self.decoder(features)
        out = self.outc(features)
        return out


class DualStreamUNet(nn.Module):
    def __init__(self, cfg):
        super(DualStreamUNet, self).__init__()
        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # stream 1 (S1)
        self.inc_s1 = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_s1 = Encoder(cfg)
        self.decoder_s1 = Decoder(cfg)

        # stream 2 (S2)
        self.inc_s2 = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_s2 = Encoder(cfg)
        self.decoder_s2 = Decoder(cfg)

        self.outc = OutConv(2*topology[0], 1)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor) -> tuple:
        # stream1 (S1)
        features_s1 = self.inc_s1(x_s1)
        features_s1 = self.encoder_s1(features_s1)
        features_s1 = self.decoder_s1(features_s1)

        # stream2 (S2)
        features_s2 = self.inc_s2(x_s2)
        features_s2 = self.encoder_s2(features_s2)
        features_s2 = self.decoder_s2(features_s2)

        x_out = torch.concat((features_s1, features_s2), dim=1)
        out = self.outc(x_out)
        return out


class ReconstructionNetV1(nn.Module):
    def __init__(self, cfg):
        super(ReconstructionNetV1, self).__init__()
        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # stream 1 (S1)
        self.inc_s1 = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_s1 = Encoder(cfg)
        self.decoder_s1 = Decoder(cfg)

        # stream 2 (S2)
        self.inc_s2 = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_s2 = Encoder(cfg)
        self.decoder_s2 = Decoder(cfg)

        self.inc_recon = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_recon = Encoder(cfg)
        self.decoder_recon = Decoder(cfg)

        self.outc = OutConv(2 * topology[0], 1)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor) -> tuple:
        # stream1 (S1)
        features_s1 = self.inc_s1(x_s1)
        features_s1 = self.encoder_s1(features_s1)
        features_s1 = self.decoder_s1(features_s1)

        # stream2 (S2)
        features_s2 = self.inc_s2(x_s2)
        features_s2 = self.encoder_s2(features_s2)
        features_s2 = self.decoder_s2(features_s2)

        # reconstruction S2 features from S1 input
        features_recon = self.inc_recon(x_s1)
        features_recon = self.encoder_s1(features_recon)
        features_s2_recon = self.decoder_s1(features_recon)

        # x_out = torch.concat((features_s1, features_s2), dim=1)
        # out = self.outc(x_out)
        return features_s1, features_s2, features_s2_recon


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs


class Decoder(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode, topology: list = None):
        super(Decoder, self).__init__()
        self.cfg = cfg

        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        # Variable scale
        n_layers = len(topology)
        up_topo = [topology[0]]  # topography upwards
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]  # last layer
            up_topo.append(out_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:

        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        return x1


# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
