import torch
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles, evaluation


def qualitative_inference_baselines(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.CHECKPOINTS.INFERENCE, cfg, device)
    net.eval()

    pred_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    pred_folder.mkdir(exist_ok=True)

    with torch.no_grad():
        for run_type in ['train', 'val', 'test']:
            ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=True)
            for item in ds:
                aoi_id = item['aoi_id']
                year = item['year']
                month = item['month']

                x_s1 = item['x_s1'].to(device)
                x_s2 = item['x_s2'].to(device)

                logits = net(x_s1.unsqueeze(0), x_s2.unsqueeze(0))
                y_pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

                transform, crs = ds.get_geo(aoi_id)
                pred_file = pred_folder / f'pred_{cfg.NAME}_{aoi_id}_{year}_{month:02d}.tif'
                geofiles.write_tif(pred_file, y_pred[:, :, None], transform, crs)


def quantitative_inference_baselines(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.TRAINER.EPOCHS, cfg, device, best_val=cfg.INFERENCE.USE_BEST_VAL)
    net.eval()

    data = {}

    for run_type in ['train', 'val', 'test']:

        data[run_type] = {}
        m_complete = evaluation.Measurer('fullmodality')
        m_incomplete = evaluation.Measurer('missingmodality')
        m_all = evaluation.Measurer('all')

        ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=True)
        for item in ds:
            x_s1 = item['x_s1'].to(device)
            x_s2 = item['x_s2'].to(device)
            y = item['y'].to(device)
            with torch.no_grad():
                logits = net(x_s1.unsqueeze(0), x_s2.unsqueeze(0))
            y_hat = torch.sigmoid(logits).squeeze().detach()

            missing_modality = item['missing_modality']
            if missing_modality:
                m_incomplete.add_sample(y, y_hat)
            else:
                m_complete.add_sample(y, y_hat)
            m_all.add_sample(y, y_hat)

        for measurer in (m_complete, m_incomplete, m_all):
            if not measurer.is_empty():
                data[run_type] = {
                    'f1': measurer.f1().item(),
                    'precision': measurer.precision().item(),
                    'recall': measurer.recall().item(),
                    'iou': measurer.iou().item(),
                    'oa': measurer.oa().item(),
                }

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'quantitative' / f'quantitative_results_{cfg.NAME}.json'
    geofiles.write_json(out_file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_inference_baselines(cfg)