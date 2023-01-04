import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baselines(net, cfg, run_type: str, epoch: float, step: int, max_samples: int = None, early_stopping: bool = False):
    net.to(device)
    net.eval()

    m_complete = metrics.MultiThresholdMetric('fullmodality', device)
    m_incomplete = metrics.MultiThresholdMetric('missingmodality', device)
    m_all = metrics.MultiThresholdMetric('all', device)

    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    max_samples = len(ds) if max_samples is None or max_samples > len(ds) else max_samples
    samples_counter = 0
    n_total = n_incomplete = 0

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
                m_complete.add_sample(y[complete_modality], y_hat[complete_modality])

            if missing_modality.any():
                m_incomplete.add_sample(y[missing_modality], y_hat[missing_modality])

            m_all.add_sample(y, y_hat)

            n_total += torch.numel(missing_modality)
            n_incomplete += torch.sum(missing_modality).item()

            samples_counter += 1
            if samples_counter == max_samples:
                break

    wandb.log({
        f'{run_type} missing_modality': n_incomplete / n_total * 100,
        'step': step, 'epoch': epoch,
    })

    return_value = None
    for measurer in (m_complete, m_incomplete, m_all):
        if not measurer.is_empty():
            f1s = measurer.compute_f1()
            precisions, recalls = measurer.precision, measurer.recall

            f1 = f1s.max().item()
            argmax_f1 = f1s.argmax()
            precision = precisions[argmax_f1].item()
            recall = recalls[argmax_f1].item()

            if measurer.name == 'all':
                return_value = f1

            suffix = 'earlystopping ' if early_stopping else ''
            wandb.log({
                suffix + f'{run_type} {measurer.name} F1': f1,
                suffix + f'{run_type} {measurer.name} precision': precision,
                suffix + f'{run_type} {measurer.name} recall': recall,
                'step': step, 'epoch': epoch,
            })

    return return_value


def proposed(net, cfg, run_type: str, epoch: float, step: int, max_samples: int = None, early_stopping: bool = False):
    net.to(device)
    net.eval()

    m_complete = metrics.MultiThresholdMetric('fullmodality', device)
    m_incomplete = metrics.MultiThresholdMetric('missingmodality', device)
    m_all = metrics.MultiThresholdMetric('all', device)

    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    max_samples = len(ds) if max_samples is None or max_samples > len(ds) else max_samples
    samples_counter = 0
    n_total = n_incomplete = 0

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
                m_complete.add_sample(y[complete_modality], y_hat_complete)
                m_all.add_sample(y[complete_modality], y_hat_complete)

            if missing_modality.any():
                features_fusion = torch.concat((features_s1, features_s2_recon), dim=1)
                logits_incomplete = net.module.outc(features_fusion[missing_modality,])
                y_hat_incomplete = torch.sigmoid(logits_incomplete).detach()
                m_incomplete.add_sample(y[missing_modality], y_hat_incomplete)
                m_all.add_sample(y[missing_modality], y_hat_incomplete)

            n_total += torch.numel(missing_modality)
            n_incomplete += torch.sum(missing_modality).item()

            samples_counter += 1
            if samples_counter == max_samples:
                break

    wandb.log({
        f'{run_type} missing_modality': n_incomplete / n_total * 100,
        'step': step, 'epoch': epoch,
    })

    return_value = None
    for measurer in (m_complete, m_incomplete, m_all):
        if not measurer.is_empty():
            f1s = measurer.compute_f1()
            precisions, recalls = measurer.precision, measurer.recall

            f1 = f1s.max().item()
            argmax_f1 = f1s.argmax()
            precision = precisions[argmax_f1].item()
            recall = recalls[argmax_f1].item()

            if measurer.name == 'all':
                return_value = f1

            suffix = 'earlystopping ' if early_stopping else ''
            wandb.log({
                suffix + f'{run_type} {measurer.name} F1': f1,
                suffix + f'{run_type} {measurer.name} precision': precision,
                suffix + f'{run_type} {measurer.name} recall': recall,
                'step': step, 'epoch': epoch,
            })

    return return_value


