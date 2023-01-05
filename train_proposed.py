import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg: experiment_manager.CfgNode):
    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sup_criterion = loss_functions.get_criterion(cfg.TRAINER.LOSS_TYPE)
    sim_criterion = loss_functions.get_criterion(cfg.RECONSTRUCTION_TRAINER.LOSS_TYPE)
    phi = cfg.RECONSTRUCTION_TRAINER.PHI

    # reset the generators
    dataset = datasets.SpaceNet7S1S2Dataset(cfg=cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val, trigger_times = 0, 0
    stop_training = False

    scaler = torch.cuda.amp.GradScaler()
    clip = 1

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set, sup_complete_loss_set, sup_incomplete_loss_set, sup_loss_set, sim_loss_set = [], [], [], [], []
        n_total = n_incomplete = 0

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x_s1 = batch['x_s1'].to(device)
            x_s2 = batch['x_s2'].to(device)
            y = batch['y'].to(device)
            missing_modality = batch['missing_modality']

            features_s1, features_s2, features_s2_recon = net(x_s1, x_s2, missing_modality)

            sup_complete_loss = sup_incomplete_loss = sim_loss = None

            complete_modality = torch.logical_not(missing_modality)
            if complete_modality.any():
                features_fusion = torch.concat((features_s1, features_s2), dim=1)
                logits_complete = net.module.outc(features_fusion[complete_modality])
                sup_complete_loss = sup_criterion(logits_complete, y[complete_modality])
                sup_complete_loss_set.append(sup_complete_loss.item())
                sim_loss = sim_criterion(features_s2[complete_modality], features_s2_recon[complete_modality])
                sim_loss = phi * sim_loss
                sim_loss_set.append(sim_loss.item())

            features_fusion = torch.concat((features_s1, features_s2_recon), dim=1)
            logits_incomplete = net.module.outc(features_fusion[missing_modality])
            sup_incomplete_loss = sup_criterion(logits_incomplete, y[missing_modality])
            sup_incomplete_loss_set.append(sup_incomplete_loss.item())

            if sup_complete_loss is None:
                sup_loss = sup_incomplete_loss
            elif sup_incomplete_loss is None:
                sup_loss = sup_complete_loss
            else:
                sup_loss = sup_complete_loss + sup_incomplete_loss
            sup_loss_set.append(sup_loss.item())

            if sim_loss is None:
                loss = sup_loss
            else:
                loss = sup_loss + sim_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

            scaler.step(optimizer)
            scaler.update()

            loss_set.append(loss.item())

            n_total += torch.numel(missing_modality)
            n_incomplete += torch.sum(missing_modality).item()
            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                time = timeit.default_timer() - start
                wandb.log({
                    'sup_complete_loss': np.mean(sup_complete_loss_set),
                    'sup_incomplete_loss': np.mean(sup_incomplete_loss_set),
                    'sup_loss': np.mean(sup_loss_set),
                    'sim_loss': np.mean(sim_loss_set) if len(sim_loss_set) > 0 else 0,
                    'loss': np.mean(loss_set),
                    'missing_percentage': n_incomplete / n_total * 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, sup_complete_loss_set, sup_incomplete_loss_set, sup_loss_set, sim_loss_set = [], [], [], [], []
                n_total = n_incomplete = 0
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)

        # evaluation at the end of an epoch
        _ = evaluation.model_evaluation(net, cfg, 'train', epoch_float, global_step)
        f1_val = evaluation.model_evaluation(net, cfg, 'val', epoch_float, global_step)
        _ = evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step)

        if cfg.EARLY_STOPPING.ENABLE:
            if f1_val <= best_f1_val:
                trigger_times += 1
                if trigger_times > cfg.EARLY_STOPPING.PATIENCE:
                    stop_training = True
            else:
                best_f1_val = f1_val
                print(f'saving network (F1 {f1_val:.3f})', flush=True)
                networks.save_checkpoint(net, optimizer, epoch, global_step, cfg, early_stopping=True)
                trigger_times = 0

        if epoch == cfg.TRAINER.EPOCHS and not cfg.DEBUG:
            print(f'saving network (end of training)', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

        if stop_training:
            break  # end of training by early stopping

    # final logging for early stopping
    if cfg.EARLY_STOPPING.ENABLE:
        net, *_ = networks.load_checkpoint(cfg.TRAINER.EPOCHS, cfg, device, best_val=True)
        _ = evaluation.model_evaluation(net, cfg, 'train', epoch_float, global_step, early_stopping=True)
        _ = evaluation.model_evaluation(net, cfg, 'val', epoch_float, global_step, early_stopping=True)
        _ = evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step, early_stopping=True)

if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        entity='population_mapping',
        project=args.project,
        tags=['missing_modality', 'urban_extraction', 'spacenet7', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

