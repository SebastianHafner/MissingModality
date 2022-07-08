import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sup_criterion = loss_functions.get_criterion(cfg.TRAINER.LOSS_TYPE)
    sim_criterion = loss_functions.get_criterion(cfg.RECONSTRUCTION_TRAINER.LOSS_TYPE)

    # reset the generators
    dataset = datasets.BuildingDataset(cfg=cfg, run_type='training')
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
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

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

            features_s1, features_s2, features_s2_recon = net(x_s1, x_s2)

            missing_modality = batch['missing_modality']
            complete_modality = torch.logical_not(missing_modality)
            n_total += torch.numel(missing_modality)
            n_incomplete += torch.sum(missing_modality).item()

            if complete_modality.any():
                features_fusion = torch.concat((features_s1, features_s2), dim=1)
                logits_complete = net.module.outc(features_fusion[complete_modality, ])
                sup_complete_loss = sup_criterion(logits_complete, y[complete_modality, ])
                sup_complete_loss_set.append(sup_complete_loss.item())

                sim_loss = sim_criterion(features_s2[complete_modality, ], features_s2_recon[complete_modality, ])
                sim_loss = (1 - cfg.RECONSTRUCTION_TRAINER.ALPHA) * sim_loss
                sim_loss_set.append(sim_loss.item())
            else:
                sup_complete_loss = 0
                sim_loss = 0

            if missing_modality.any():
                features_fusion = torch.concat((features_s1, features_s2_recon), dim=1)
                logits_incomplete = net.module.outc(features_fusion[missing_modality, ])
                sup_incomplete_loss = sup_criterion(logits_incomplete, y[missing_modality, ])
                sup_incomplete_loss_set.append(sup_incomplete_loss.item())
            else:
                sup_incomplete_loss = 0

            sup_loss = sup_complete_loss + sup_incomplete_loss
            sup_loss = cfg.RECONSTRUCTION_TRAINER.ALPHA * sup_loss

            loss = sup_loss + sim_loss
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if cfg.DEBUG:
                break

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation_missingmodality(net, cfg, device, 'training', epoch_float, global_step)
                evaluation.model_evaluation_missingmodality(net, cfg, device, 'validation', epoch_float, global_step)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'sup_loss': np.mean(sup_loss_set),
                    'sim_loss': np.mean(sim_loss_set) if len(sim_loss_set) > 0 else 0,
                    'loss': np.mean(loss_set),
                    'missing_percentage': n_incomplete / n_total * 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                n_total = n_incomplete = 0
                loss_set, sup_loss_set, sim_loss_set = [], [], []
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        evaluation.model_evaluation_missingmodality(net, cfg, device, 'training', epoch_float, global_step)
        evaluation.model_evaluation_missingmodality(net, cfg, device, 'test', epoch_float, global_step)

        if epoch in save_checkpoints:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)


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

