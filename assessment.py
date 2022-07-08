import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import experiment_manager, networks, datasets, metrics, geofiles, parsers
from sklearn.metrics import precision_recall_curve, auc


def quantitative_assessment_semantic(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)

    y_trues, y_preds = [], []

    with torch.no_grad():
        for item in tqdm(ds):
            # semantic labels
            gt_t1, gt_t2 = item['y_sem_t1'].to(device), item['y_sem_t2'].to(device)
            y_trues.extend([gt_t1.flatten(), gt_t2.flatten()])

            # semantic predictions
            x_t1, x_t2 = item['x_t1'].to(device), item['x_t2'].to(device)
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            if cfg.MODEL.TYPE == 'dtsiameseunet':
                _, logits_t1, logits_t2 = logits
            else:
                logits_t1, logits_t2 = logits[5:]
            pred_t1, pred_t2 = torch.sigmoid(logits_t1), torch.sigmoid(logits_t2)
            y_preds.extend([pred_t1.flatten(), pred_t2.flatten()])

    y_preds = torch.cat(y_preds).flatten().cpu().numpy()
    y_trues = torch.cat(y_trues).flatten().cpu().numpy()

    file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'quantitative_results_semantic_{run_type}.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)

    f1 = metrics.f1_score_from_prob(y_preds, y_trues)
    precision = metrics.precsision_from_prob(y_preds, y_trues)
    recall = metrics.recall_from_prob(y_preds, y_trues)
    precisions, recalls, _ = precision_recall_curve(y_trues, y_preds)
    auc_pr = auc(recalls, precisions)

    data[cfg.NAME] = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc_pr,
    }

    geofiles.write_json(file, data)


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


def qualitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits_change = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits_change).squeeze().detach()

        gt_change = item['y_change'].squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(x_t1.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])
        axs[1, 0].imshow(x_t2.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])

        axs[0, 1].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 1].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME / f'change_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def qualitative_assessment_dualtask(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits[0]).squeeze().detach()
        gt_change = item['y_change'].squeeze()

        logits_stream1_sem_t1, logits_stream1_sem_t2, logits_stream2_sem_t1, logits_stream2_sem_t2 = logits[3:]

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        # s2 images

        axs[0, 0].imshow(s2_t1.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])
        axs[1, 0].imshow(s2_t2.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])

        axs[0, 3].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 3].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME / f'change_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def qualitative_comparison(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits_change = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits_change).squeeze().detach()

        gt_change = item['y_change'].squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(x_t1.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])
        axs[1, 0].imshow(x_t2.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])

        axs[0, 1].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 1].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME / f'change_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment_semantic(cfg, run_type='test')
