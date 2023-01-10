import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from pathlib import Path

from utils import experiment_manager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_network(cfg: experiment_manager.CfgNode):
    if cfg.MODEL.TYPE == 'unet':
        model = UNet(cfg)
    elif cfg.MODEL.TYPE == 'dualstreamunet':
        model = DualStreamUNet(cfg)
    elif cfg.MODEL.TYPE == 'dualstreamunetplus':
        model = DualStreamUNetPlus(cfg)
    elif cfg.MODEL.TYPE == 'reconstructionnet_v1':
        model = ReconstructionNetV1(cfg)
    elif cfg.MODEL.TYPE == 'reconstructionnet_v1_finetuned':
        model = ReconstructionNetV1Finetuned(cfg)
    elif cfg.MODEL.TYPE == 'mmtmdualstreamunet':
        model = MMTM_DSUNet(cfg)
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    return nn.DataParallel(model)


def save_checkpoint(network, optimizer, epoch, step, cfg: experiment_manager.CfgNode, early_stopping: bool = False):
    if early_stopping:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_early_stopping.pt'
    else:
        save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch: float, cfg: experiment_manager.CfgNode, device: torch.device, net_file: Path = None,
                    best_val: bool = False):
    net = create_network(cfg)
    net.to(device)

    if net_file is None:
        net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    if best_val:
        net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_early_stopping.pt'

    checkpoint = torch.load(net_file, map_location=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.cfg = cfg
        self.requires_missing_modality = False
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
        self.requires_missing_modality = False
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


class DualStreamUNetPlus(nn.Module):
    def __init__(self, cfg):
        super(DualStreamUNetPlus, self).__init__()
        self.cfg = cfg
        self.requires_missing_modality = True
        topology = cfg.MODEL.TOPOLOGY

        # stream 1 (S1)
        self.inc_s1 = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_s1 = Encoder(cfg)
        self.decoder_s1 = Decoder(cfg)

        # stream 2 (S2)
        self.inc_s2 = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_s2 = Encoder(cfg)
        self.decoder_s2 = Decoder(cfg)

        self.outc = OutConv(2 * topology[0], 1)

        # parameters
        patch_size = cfg.AUGMENTATION.CROP_SIZE
        self.ravg_features_s2 = nn.Parameter(torch.zeros((1, topology[0], patch_size, patch_size), requires_grad=False))
        self.n = nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor, missing_modality: torch.tensor) -> tuple:
        # stream1 (S1)
        features_s1 = self.inc_s1(x_s1)
        features_s1 = self.encoder_s1(features_s1)
        features_s1 = self.decoder_s1(features_s1)

        # stream2 (S2)
        features_s2 = self.inc_s2(x_s2)
        features_s2 = self.encoder_s2(features_s2)
        features_s2 = self.decoder_s2(features_s2)

        if missing_modality.any():
            features_s2[missing_modality] = self.ravg_features_s2

        x_out = torch.concat((features_s1, features_s2), dim=1)
        out = self.outc(x_out)

        # update ravg s2 features
        complete_modality = torch.logical_not(missing_modality)
        if self.training and complete_modality.any():
            nominator = self.ravg_features_s2 * self.n + torch.sum(features_s2[complete_modality], dim=0)
            demoninator = self.n + torch.sum(complete_modality)
            self.ravg_features_s2 = nn.Parameter(nominator / demoninator, requires_grad=False)
            self.n = nn.Parameter(self.n + torch.sum(complete_modality).item(), requires_grad=False)
        return out


class ReconstructionNetV1(nn.Module):
    def __init__(self, cfg):
        super(ReconstructionNetV1, self).__init__()
        self.cfg = cfg
        self.requires_missing_modality = True
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

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor, missing_modality: torch.tensor) -> tuple:
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

        if self.training:
            return features_s1, features_s2, features_s2_recon
        else:
            # replacing the s2 features for missing modality samples with the reconstructed features
            if missing_modality.any():
                features_s2[missing_modality] = features_s2_recon
            x_out = torch.concat((features_s1, features_s2), dim=1)
            out = self.outc(x_out)
            return out


class ReconstructionNetV1Finetuned(nn.Module):
    def __init__(self, cfg):
        super(ReconstructionNetV1Finetuned, self).__init__()
        self.cfg = cfg
        self.requires_missing_modality = True
        topology = cfg.MODEL.TOPOLOGY

        # stream 1 (S1)
        branch_cfg = experiment_manager.setup_cfg_manual(cfg.MODEL.PRETRAINED_BRANCH, Path(cfg.PATHS.OUTPUT),
                                                         Path(cfg.PATHS.DATASET))
        pretrained_branch, *_ = load_checkpoint(None, branch_cfg, device, best_val=True)
        if cfg.MODEL.FREEZE_PRETRAINED_BRANCH:
            for param in pretrained_branch.module.parameters():
                param.requires_grad = False
        self.inc_s1 = pretrained_branch.module.inc
        self.encoder_s1 = pretrained_branch.module.encoder
        self.decoder_s1 = pretrained_branch.module.decoder

        # stream 2 (S2)
        self.inc_s2 = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_s2 = Encoder(cfg)
        self.decoder_s2 = Decoder(cfg)

        self.inc_recon = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_recon = Encoder(cfg)
        self.decoder_recon = Decoder(cfg)

        self.outc = OutConv(2 * topology[0], 1)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor, missing_modality: torch.tensor) -> tuple:
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

        if self.training:
            return features_s1, features_s2, features_s2_recon
        else:
            # replacing the s2 features for missing modality samples with the reconstructed features
            if missing_modality.any():
                features_s2[missing_modality] = features_s2_recon
            x_out = torch.concat((features_s1, features_s2), dim=1)
            out = self.outc(x_out)
            return out


class MMTM_DSUNet(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(MMTM_DSUNet, self).__init__()

        self.cfg = cfg
        self.requires_missing_modality = False

        self.inc_s1 = InConv(len(cfg.DATALOADER.S1_BANDS), 64, DoubleConv)
        self.inc_s2 = InConv(len(cfg.DATALOADER.S2_BANDS), 64, DoubleConv)

        self.mmtm1 = self.mmtm1 = MMTM(64, 64)

        self.max1_s1 = nn.MaxPool2d(2)
        self.max1_s2 = nn.MaxPool2d(2)

        self.conv1_s1 = DoubleConv(64, 128)
        self.conv1_s2 = DoubleConv(64, 128)

        self.mmtm2 = MMTM(128, 128)

        self.max2_s1 = nn.MaxPool2d(2)
        self.max2_s2 = nn.MaxPool2d(2)

        self.conv2_s1 = DoubleConv(128, 128)
        self.conv2_s2 = DoubleConv(128, 128)

        self.mmtm3 = MMTM(128, 128)

        self.up1_s1 = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))
        self.up1_s2 = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))

        self.conv3_s1 = DoubleConv(256, 64)
        self.conv3_s2 = DoubleConv(256, 64)

        self.mmtm4 = MMTM(64, 64)

        self.up2_s1 = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))
        self.up2_s2 = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))

        self.conv4_s1 = DoubleConv(128, 64)
        self.conv4_s2 = DoubleConv(128, 64)

        self.outc_s1 = OutConv(64, 1)
        self.outc_s2 = OutConv(64, 1)

        self.mmtm_units = [self.mmtm1, self.mmtm2, self.mmtm3, self.mmtm4]

    def forward(self, x_s1: torch.tensor, x_s2: torch.tensor, rec_mmtm_squeeze: bool = False,
                mmtm_off: bool = False):
        mmtm_kwargs = {
            'turnoff_cross_modal_flow': True if mmtm_off else False,
            'rec_mmtm_squeeze': rec_mmtm_squeeze,
        }

        features_s1 = self.inc_s1(x_s1)
        features_s2 = self.inc_s2(x_s2)

        features_s1, features_s2 = self.mmtm1(features_s1, features_s2, **mmtm_kwargs)
        skip1_s1, skip1_s2 = features_s1, features_s2

        features_s1 = self.max1_s1(features_s1)
        features_s1 = self.conv1_s1(features_s1)
        features_s2 = self.max1_s2(features_s2)
        features_s2 = self.conv1_s2(features_s2)

        features_sar, features_opt = self.mmtm2(features_s1, features_s2, **mmtm_kwargs)
        skip2_s1, skip2_s2 = features_s1, features_s2

        features_s1 = self.max2_s1(features_s1)
        features_s1 = self.conv2_s1(features_s1)
        features_s2 = self.max2_s2(features_s2)
        features_s2 = self.conv2_s2(features_s2)

        features_s1, features_s2 = self.mmtm3(features_s1, features_s2, **mmtm_kwargs)

        features_s1 = self.up1_s1(features_s1)
        diffY = skip2_s1.detach().size()[2] - features_s1.detach().size()[2]
        diffX = skip2_s1.detach().size()[3] - features_s1.detach().size()[3]
        features_s1 = F.pad(features_s1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_s1 = torch.cat([skip2_s1, features_s1], dim=1)
        features_s1 = self.conv3_s1(features_s1)

        features_s2 = self.up1_s2(features_s2)
        features_s2 = F.pad(features_s2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_s2 = torch.cat([skip2_s2, features_s2], dim=1)
        features_s2 = self.conv3_s2(features_s2)

        features_s1, features_s2 = self.mmtm4(features_s1, features_s2, **mmtm_kwargs)

        features_s1 = self.up2_s1(features_s1)
        diffY = skip1_s1.detach().size()[2] - features_s1.detach().size()[2]
        diffX = skip1_s1.detach().size()[3] - features_s1.detach().size()[3]
        features_s1 = F.pad(features_s1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_s1 = torch.cat([skip1_s1, features_s1], dim=1)
        features_s1 = self.conv4_s1(features_s1)

        features_s2 = self.up2_s2(features_s2)
        features_s2 = F.pad(features_s2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_s2 = torch.cat([skip1_s2, features_s2], dim=1)
        features_s2 = self.conv4_s2(features_s2)

        out_s1 = self.outc_s1(features_s1)
        out_s2 = self.outc_s2(features_s2)
        return out_s1, out_s2

    def mmtm_squeeze_features_recorded(self):
        return torch.all(torch.tensor([mmtm.avg_squeeze_recorded() for mmtm in self.mmtm_units]))


class MMTM(nn.Module):
    def __init__(self, dim_s1, dim_s2):
        super(MMTM, self).__init__()

        dim = dim_s1 + dim_s2
        dim_out = int(2 * dim)

        # self.ravg_out_s1 = torch.zeros(dim_s1).to(device)
        # self.ravg_out_s2 = torch.zeros(dim_s2).to(device)
        # self.step = 0

        # self.ravg_squeeze_s1 = nn.Parameter(torch.zeros((1, dim_s1), requires_grad=False))
        # self.ravg_squeeze_s2 = nn.Parameter(torch.zeros((1, dim_s2), requires_grad=False))
        # self.rec_step = 0

        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_s1 = nn.Linear(dim_out, dim_s1)
        self.fc_s2 = nn.Linear(dim_out, dim_s2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, features_s1: torch.tensor, features_s2: torch.tensor, turnoff_cross_modal_flow: bool = False,
                rec_mmtm_squeeze: bool = False):

        tview_s1 = features_s1.view(features_s1.shape[:2] + (-1,))
        squeeze_s1 = torch.mean(tview_s1, dim=-1)

        tview_s2 = features_s2.view(features_s2.shape[:2] + (-1,))
        squeeze_s2 = torch.mean(tview_s2, dim=-1)

        # if rec_mmtm_squeeze:
        #     ravg_squeeze_s1 = (squeeze_s1.mean(0) + self.ravg_squeeze_s1 * self.rec_step) / (self.rec_step + 1)
        #     self.ravg_squeeze_s1 = nn.Parameter(ravg_squeeze_s1.detach())
        #     ravg_squeeze_s2 = (squeeze_s2.mean(0) + self.ravg_squeeze_s2 * self.rec_step) / (self.rec_step + 1)
        #     self.ravg_squeeze_s2 = nn.Parameter(ravg_squeeze_s2.detach())
        #     self.rec_step += 1
        #
        # if not turnoff_cross_modal_flow:
        squeeze = torch.cat((squeeze_s1, squeeze_s2), 1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        s1_out = self.fc_s1(excitation)
        s2_out = self.fc_s2(excitation)
        # else:
        #     ravg_squeeze_s2 = self.ravg_squeeze_s2.expand(features_s1.shape[0], -1)
        #     squeeze_unimodal_s1 = torch.cat([squeeze_s1, ravg_squeeze_s2], 1)
        #     excitation_unimodal_s1 = self.relu(self.fc_squeeze(squeeze_unimodal_s1))
        #     s1_out = self.fc_s1(excitation_unimodal_s1)
        #
        #     ravg_squeeze_s1 = self.ravg_squeeze_s1.expand(features_s2.shape[0], -1)
        #     squeeze_unimodal_s2 = torch.cat([ravg_squeeze_s1, squeeze_s2], 1)
        #     excitation_unimodal_s2 = self.relu(self.fc_squeeze(squeeze_unimodal_s2))
        #     s2_out = self.fc_s2(excitation_unimodal_s2)

        s1_out = self.sigmoid(s1_out)
        s2_out = self.sigmoid(s2_out)

        # running average weights of output
        # self.ravg_out_s1 = (s1_out.mean(0) + self.ravg_out_s1 * self.step).detach() / (self.step + 1)
        # self.ravg_out_s2 = (s2_out.mean(0) + self.ravg_out_s2 * self.step).detach() / (self.step + 1)
        # self.step += 1

        # matching the shape of the excitation signals to the input features for recalibration
        # (B, C) -> (B, C, H, W)
        s1_out = s1_out.view(s1_out.shape + (1,) * (len(features_s1.shape) - len(s1_out.shape)))
        s2_out = s2_out.view(s2_out.shape + (1,) * (len(features_s2.shape) - len(s2_out.shape)))

        return features_s1 * s1_out, features_s2 * s2_out

    # def avg_squeeze_recorded(self):
    #     if torch.sum(self.ravg_squeeze_s1) + torch.sum(self.ravg_squeeze_s2) == 0:
    #         return False
    #     else:
    #         return True


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
