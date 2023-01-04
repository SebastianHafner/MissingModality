import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Measurer(object):
    def __init__(self, name: str = None, threshold: float = 0.5):

        self.name = name
        self.threshold = threshold

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self._precision = None
        self._recall = None

        self.eps = 10e-05

    def add_sample(self, y: torch.Tensor, y_hat: torch.Tensor):
        y = y.bool()
        y_hat = y_hat > self.threshold

        self.TP += torch.sum(y & y_hat).float()
        self.TN += torch.sum(~y & ~y_hat).float()
        self.FP += torch.sum(y_hat & ~y).float()
        self.FN += torch.sum(~y_hat & y).float()

    def precision(self):
        if self._precision is None:
            self._precision = self.TP / (self.TP + self.FP + self.eps)
        return self._precision

    def recall(self):
        if self._recall is None:
            self._recall = self.TP / (self.TP + self.FN + self.eps)
        return self._recall

    def compute_basic_metrics(self):
        false_pos_rate = self.FP / (self.FP + self.TN + self.eps)
        false_neg_rate = self.FN / (self.FN + self.TP + self.eps)
        return false_pos_rate, false_neg_rate

    def f1(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall() + self.eps)

    def iou(self):
        return self.TP / (self.TP + self.FP + self.FN + self.eps)

    def oa(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + self.eps)

    def is_empty(self):
        return True if (self.TP + self.TN + self.FP + self.FN) == 0 else False


def baselines(net, cfg, run_type: str, epoch: float, step: int, max_samples: int = None, early_stopping: bool = False):
    net.to(device)
    net.eval()

    m_complete, m_incomplete, m_all = Measurer('fullmodality'), Measurer('missingmodality'), Measurer('all')

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
            f1 = measurer.f1()
            false_pos_rate, false_neg_rate = measurer.compute_basic_metrics()

            suffix = 'earlystopping ' if early_stopping else ''
            wandb.log({
                suffix + f'{run_type} {measurer.name} F1': measurer.f1(),
                suffix + f'{run_type} {measurer.name} fpr': false_pos_rate,
                suffix + f'{run_type} {measurer.name} fnr': false_neg_rate,
                'step': step, 'epoch': epoch,
            })

            if measurer.name == 'all':
                return_value = f1

    return return_value


def proposed(net, cfg, run_type: str, epoch: float, step: int, max_samples: int = None, early_stopping: bool = False):
    net.to(device)
    net.eval()

    m_complete, m_incomplete, m_all = Measurer('fullmodality'), Measurer('missingmodality'), Measurer('all')

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
            f1 = measurer.f1()
            false_pos_rate, false_neg_rate = measurer.compute_basic_metrics()

            suffix = 'earlystopping ' if early_stopping else ''
            wandb.log({
                suffix + f'{run_type} {measurer.name} F1': measurer.f1(),
                suffix + f'{run_type} {measurer.name} fpr': false_pos_rate,
                suffix + f'{run_type} {measurer.name} fnr': false_neg_rate,
                'step': step, 'epoch': epoch,
            })

            if measurer.name == 'all':
                return_value = f1

    return return_value


