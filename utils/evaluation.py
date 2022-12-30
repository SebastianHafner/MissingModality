import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baselines(net, cfg, run_type: str, epoch: float, step: int, max_samples: int = None, early_stopping: bool = False):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_complete = metrics.MultiThresholdMetric(thresholds)
    measurer_incomplete = metrics.MultiThresholdMetric(thresholds)
    measurer_all = metrics.MultiThresholdMetric(thresholds)

    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    max_samples = len(ds) if max_samples is None or max_samples > len(ds) else max_samples
    samples_counter = 0

    with torch.no_grad():
        for step, item in enumerate(dataloader):
            x_s1 = item['x_s1'].to(device)
            x_s2 = item['x_s2'].to(device)
            y = item['y'].to(device)

            logits = net(x_s1, x_s2)
            y_hat = torch.sigmoid(logits).detach()

            missing_modality = item['missing_modality']
            complete_modality = torch.logical_not(missing_modality)

            if complete_modality.any():
                measurer_complete.add_sample(y[complete_modality], y_hat[complete_modality])

            if missing_modality.any():
                measurer_incomplete.add_sample(y[missing_modality], y_hat[missing_modality])

            measurer_all.add_sample(y, y_hat)

            samples_counter += 1
            if samples_counter == max_samples:
                break

    for measurer, name in zip((measurer_complete, measurer_incomplete, measurer_all),
                              ['fullmodality', 'missingmodality', 'all']):
        if not measurer.is_empty():
            f1s = measurer.compute_f1()
            precisions, recalls = measurer.precision, measurer.recall

            f1 = f1s.max().item()
            argmax_f1 = f1s.argmax()
            precision = precisions[argmax_f1].item()
            recall = recalls[argmax_f1].item()
            if early_stopping:
                wandb.log({
                    f'earlystopping {run_type} {name} F1': f1,
                    f'earlystopping {run_type} {name} precision': precision,
                    f'earlystopping {run_type} {name} recall': recall,
                    'step': step, 'epoch': epoch,
                })
            else:
                wandb.log({
                    f'{run_type} {name} F1': f1,
                    f'{run_type} {name} precision': precision,
                    f'{run_type} {name} recall': recall,
                    'step': step, 'epoch': epoch,
                })


def proposed(net, cfg, run_type: str, epoch: float, step: int, max_samples: int = None, early_stopping: bool = False):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_complete = metrics.MultiThresholdMetric(thresholds)
    measurer_incomplete = metrics.MultiThresholdMetric(thresholds)
    measurer_all = metrics.MultiThresholdMetric(thresholds)

    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    max_samples = len(ds) if max_samples is None or max_samples > len(ds) else max_samples
    samples_counter = 0

    with torch.no_grad():
        for step, item in enumerate(dataloader):
            x_s1 = item['x_s1'].to(device)
            x_s2 = item['x_s2'].to(device)
            y = item['y'].to(device)

            features_s1, features_s2, features_s2_recon = net(x_s1, x_s2)

            missing_modality = item['missing_modality']
            complete_modality = torch.logical_not(missing_modality)

            if complete_modality.any():
                features_fusion = torch.concat((features_s1, features_s2), dim=1)
                logits_complete = net.module.outc(features_fusion[complete_modality,])
                y_hat_complete = torch.sigmoid(logits_complete).detach()
                measurer_complete.add_sample(y[complete_modality], y_hat_complete)
                measurer_all.add_sample(y[complete_modality], y_hat_complete)

            if missing_modality.any():
                features_fusion = torch.concat((features_s1, features_s2_recon), dim=1)
                logits_incomplete = net.module.outc(features_fusion[missing_modality,])
                y_hat_incomplete = torch.sigmoid(logits_incomplete).detach()
                measurer_incomplete.add_sample(y[missing_modality], y_hat_incomplete)
                measurer_all.add_sample(y[missing_modality], y_hat_incomplete)

            samples_counter += 1
            if samples_counter == max_samples:
                break

    for measurer, name in zip((measurer_complete, measurer_incomplete, measurer_all),
                              ['fullmodality', 'missingmodality', 'all']):
        if not measurer.is_empty():
            f1s = measurer.compute_f1()
            precisions, recalls = measurer.precision, measurer.recall

            f1 = f1s.max().item()
            argmax_f1 = f1s.argmax()
            precision = precisions[argmax_f1].item()
            recall = recalls[argmax_f1].item()

            if early_stopping:
                wandb.log({
                    f'earlystopping {run_type} {name} F1': f1,
                    f'earlystopping {run_type} {name} precision': precision,
                    f'earlystopping {run_type} {name} recall': recall,
                    'step': step, 'epoch': epoch,
                })
            else:
                wandb.log({
                    f'{run_type} {name} F1': f1,
                    f'{run_type} {name} precision': precision,
                    f'{run_type} {name} recall': recall,
                    'step': step, 'epoch': epoch,
                })



