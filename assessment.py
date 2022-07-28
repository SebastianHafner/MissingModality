import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import experiment_manager, networks, datasets, metrics, geofiles, parsers
import matplotlib.pyplot as plt


def quantitative_assessment_fullmodality(cfg: experiment_manager.CfgNode, run_type: str = 'validation'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.CHECKPOINTS.INFERENCE, cfg, device)
    net.eval()
    ds = datasets.BuildingDataset(cfg, run_type, no_augmentations=True)

    y_trues_complete, y_trues_incomplete, y_trues_all = [], [], []
    y_preds_complete, y_preds_incomplete, y_preds_all = [], [], []

    with torch.no_grad():
        for item in tqdm(ds):
            x_s1 = item['x_s1'].to(device)
            x_s2 = item['x_s2'].to(device)

            logits = net(x_s1.unsqueeze(0), x_s2.unsqueeze(0))
            pred = torch.sigmoid(logits).detach()

            gt = item['y'].to(device)

            y_trues_all.append(gt.flatten())
            y_preds_all.append(pred.flatten())

            missing_modality = item['missing_modality']
            if missing_modality:
                y_trues_incomplete.append(gt.flatten())
                y_preds_incomplete.append(pred.flatten())
            else:
                y_trues_complete.append(gt.flatten())
                y_preds_complete.append(pred.flatten())

    y_trues_all = torch.cat(y_trues_all).cpu().numpy()
    y_preds_all = torch.cat(y_preds_all).cpu().numpy()

    y_trues_incomplete = torch.cat(y_trues_incomplete).cpu().numpy()
    y_preds_incomplete = torch.cat(y_preds_incomplete).cpu().numpy()

    y_trues_complete = torch.cat(y_trues_complete).cpu().numpy()
    y_preds_complete = torch.cat(y_preds_complete).cpu().numpy()


    file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'quantitative_results_{run_type}.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)
    data[cfg.NAME] = {}

    for y_trues, y_preds, name in zip([y_trues_all, y_trues_incomplete, y_trues_complete],
                                      [y_preds_all, y_preds_incomplete, y_preds_complete],
                                      ['all', 'missingmodality', 'fullmodality']):

            if not y_trues.size == 0:
                f1 = metrics.f1_score_from_prob(y_preds, y_trues)
                iou = metrics.iou_from_prob(y_preds, y_trues)
                oa = metrics.oa_from_prob(y_preds, y_trues)

                data[cfg.NAME][name] = {
                    'f1': f1,
                    'iou': iou,
                    'oa': oa,
                }

    geofiles.write_json(file, data)


def quantitative_assessment_missingmodality(cfg: experiment_manager.CfgNode, run_type: str = 'validation'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.CHECKPOINTS.INFERENCE, cfg, device)
    net.eval()
    ds = datasets.BuildingDataset(cfg, run_type, no_augmentations=True)

    y_trues_complete, y_trues_incomplete, y_trues_all = [], [], []
    y_preds_complete, y_preds_incomplete, y_preds_all = [], [], []

    with torch.no_grad():
        for item in tqdm(ds):
            x_s1 = item['x_s1'].to(device)
            x_s2 = item['x_s2'].to(device)

            features_s1, features_s2, features_s2_recon = net(x_s1.unsqueeze(0), x_s2.unsqueeze(0))

            missing_modality = item['missing_modality']
            if missing_modality:
                features_fusion = torch.concat((features_s1, features_s2_recon), dim=1)
            else:
                features_fusion = torch.concat((features_s1, features_s2), dim=1)

            logits = net.module.outc(features_fusion)
            pred = torch.sigmoid(logits).detach()

            gt = item['y'].to(device)

            y_trues_all.append(gt.flatten())
            y_preds_all.append(pred.flatten())

            if missing_modality:
                y_trues_incomplete.append(gt.flatten())
                y_preds_incomplete.append(pred.flatten())
            else:
                y_trues_complete.append(gt.flatten())
                y_preds_complete.append(pred.flatten())

    y_trues_all = torch.cat(y_trues_all).cpu().numpy()
    y_preds_all = torch.cat(y_preds_all).cpu().numpy()

    y_trues_incomplete = torch.cat(y_trues_incomplete).cpu().numpy()
    y_preds_incomplete = torch.cat(y_preds_incomplete).cpu().numpy()

    y_trues_complete = torch.cat(y_trues_complete).cpu().numpy()
    y_preds_complete = torch.cat(y_preds_complete).cpu().numpy()

    file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'quantitative_results_{run_type}.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)
    data[cfg.NAME] = {}

    for y_trues, y_preds, name in zip([y_trues_all, y_trues_incomplete, y_trues_complete],
                                      [y_preds_all, y_preds_incomplete, y_preds_complete],
                                      ['all', 'missingmodality', 'fullmodality']):

        if not y_trues.size == 0:
            f1 = metrics.f1_score_from_prob(y_preds, y_trues)
            iou = metrics.iou_from_prob(y_preds, y_trues)
            oa = metrics.oa_from_prob(y_preds, y_trues)

            data[cfg.NAME][name] = {
                'f1': f1,
                'iou': iou,
                'oa': oa,
            }

    geofiles.write_json(file, data)


def qualitative_assessment(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.CHECKPOINTS.INFERENCE, cfg, device)
    net.eval()

    plot_folder = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME
    plot_folder.mkdir(exist_ok=True)

    with torch.no_grad():
        for run_type in ['training', 'test']:
            ds = datasets.BuildingDataset(cfg, run_type, no_augmentations=True)
            for item in ds:
                aoi_id = item['aoi_id']
                year = item['year']
                month = item['month']

                x_s1 = item['x_s1'].to(device).unsqueeze(0)
                x_s2 = item['xs2'].to(device).unsqueeze(0)
                y = item['y'].squeeze()

                if cfg.MODALITY == 'full':
                    logits = net(x_s1, x_s2)
                else:
                    features_s1, features_s2, features_s2_recon = net(x_s1, x_s2)
                    missing_modality = item['missing_modality']
                    if missing_modality:
                        features_fusion = torch.concat((features_s1, features_s2_recon), dim=1)
                    else:
                        features_fusion = torch.concat((features_s1, features_s2), dim=1)
                    logits = net.module.outc(features_fusion)
                y_pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(y, cmap='jet')
                axs[1].imshow(y_pred, cmap='jet')
                for _, ax in np.ndenumerate(axs):
                    ax.set_xticks([])
                    ax.set_yticks([])

                pred_file = plot_folder / f'{run_type}_{cfg.NAME}_{aoi_id}_{year}_{month:02d}.png'
                plt.savefig(pred_file, dpi=300, bbox_inches='tight')
                plt.close(fig)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # quantitative_assessment_fullmodality(cfg, run_type='validation')
    # quantitative_assessment_missingmodality(cfg, run_type='validation')
    qualitative_assessment(cfg)