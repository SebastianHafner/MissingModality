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


def run_training(cfg: experiment_manager.CfgNode):
    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.TRAINER.LOSS_TYPE)

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
    save_checkpoints = cfg.CHECKPOINTS.SAVE
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    evaluation.model_evaluation_baselines(net, cfg, device, 'training', epoch_float, global_step,
                                          cfg.LOGGING.EPOCH_MAX_SAMPLES)
    evaluation.model_evaluation_baselines(net, cfg, device, 'validation', epoch_float, global_step,
                                          cfg.LOGGING.EPOCH_MAX_SAMPLES)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []
        n_total = n_incomplete = 0

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x_s1 = batch['x_s1'].to(device)
            x_s2 = batch['x_s2'].to(device)
            y = batch['y'].to(device)

            missing_modality = batch['missing_modality']

            logits = net(x_s1, x_s2)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            n_total += torch.numel(missing_modality)
            n_incomplete += torch.sum(missing_modality).item()
            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOGGING.STEP_FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                # evaluation on sample of training and validation set
                evaluation.model_evaluation_baselines(net, cfg, device, 'training', epoch_float, global_step,
                                                      cfg.LOGGING.STEP_MAX_SAMPLES)
                evaluation.model_evaluation_baselines(net, cfg, device, 'validation', epoch_float, global_step,
                                                      cfg.LOGGING.STEP_MAX_SAMPLES)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'missing_percentage': n_incomplete / n_total * 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []
                n_total = n_incomplete = 0
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        if epoch_float % cfg.LOGGING.EPOCH_FREQUENCY == 0:
            # evaluation at the end of an epoch
            evaluation.model_evaluation_baselines(net, cfg, device, 'training', epoch_float, global_step,
                                                  cfg.LOGGING.EPOCH_MAX_SAMPLES)
            evaluation.model_evaluation_baselines(net, cfg, device, 'validation', epoch_float, global_step,
                                                  cfg.LOGGING.EPOCH_MAX_SAMPLES)

        if epoch in save_checkpoints and not cfg.DEBUG:
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
