import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import experiment_manager, networks, datasets, metrics, geofiles, parsers


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


def qualitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'validation'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.CHECKPOINTS.INFERENCE, cfg, device)
    net.eval()
    ds = datasets.BuildingDataset(cfg, run_type, no_augmentations=True)

    y_trues, y_preds = [], []

    with torch.no_grad():
        for item in tqdm(ds):
            x_s1 = item['x_s1'].to(device)
            x_s2 = item['x_s2'].to(device)

            logits = net(x_s1.unsqueeze(0), x_s2.unsqueeze(0))
            pred = torch.sigmoid(logits)

            gt = item['y'].to(device)



def qualitative_comparison(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)

    plot_folder = Path(cfg.PATHS.OUTPUT) / 'plots' / 'qualitative_comparison'
    plot_folder.mkdir(exist_ok=True)

    inference_folder = Path(cfg.PATHS.OUTPUT) / 'inference'

    def plot_baseline(ax, baseline: str, aoi_id: str):
        pred_file = inference_folder / baseline / f'pred_{baseline}_{aoi_id}.tif'
        pred, *_ = geofiles.read_tif(pred_file)
        pred = pred > 0.5
        ax.imshow(pred, cmap='gray')

    with torch.no_grad():
        for item in ds:
            aoi_id = item['aoi_id']
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            logits_change = logits[0] if cfg.MODEL.TYPE == 'whatevernet3' else logits
            y_pred_change = torch.sigmoid(logits_change).squeeze().detach().cpu()
            gt_change = item['y_change'].squeeze()

            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            plt.tight_layout()

            rgb_t1 = ds.load_s2_rgb(aoi_id, item['year_t1'], item['month_t1'])
            axs[0, 0].imshow(rgb_t1)
            rgb_t2 = ds.load_s2_rgb(aoi_id, item['year_t2'], item['month_t2'])
            axs[0, 1].imshow(rgb_t2)
            axs[0, 2].imshow(gt_change.numpy(), cmap='gray')

            axs[1, 2].imshow(y_pred_change.numpy() > 0.5, cmap='gray')

            plot_baseline(axs[1, 0], 'baseline_dualstream_gamma', aoi_id)
            plot_baseline(axs[1, 1], 'siamesedt_gamma', aoi_id)

            index = 0
            for _, ax in np.ndenumerate(axs):
                char = chr(ord('a') + index)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xlabel(f'({char})', fontsize=16, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                index += 1

            out_file = plot_folder / f'qualitative_comparison_{aoi_id}.png'
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close(fig)




if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment_fullmodality(cfg, run_type='validation')
