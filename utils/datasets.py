import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles


class AbstractBuildingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type
        self.root_path = Path(cfg.PATHS.DATASET)

        self.metadata = geofiles.load_json(self.root_path / f'metadata.json')

        self.s1_band_indices = cfg.DATALOADER.S1_BANDS
        self.s2_band_indices = cfg.DATALOADER.S2_BANDS

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_s1_img(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s1_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_s2_img(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s2_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label > 0
        return np.nan_to_num(label).astype(np.float32)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def get_geo(self, aoi_id: str) -> tuple:
        timestamps = self.metadata[aoi_id]
        timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1']]
        year, month = timestamps[0]
        file = self.root_path / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
        _, transform, crs = geofiles.read_tif(file)
        return transform, crs

    def load_s2_rgb(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, [2, 1, 0]] / 0.3, 0, 1)
        return np.nan_to_num(img).astype(np.float32)


# dataset for urban extraction with building footprints
class BuildingDataset(AbstractBuildingDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg, run_type)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        if run_type == 'training':
            self.aoi_ids = list(cfg.DATASET.TRAINING_IDS)
        else:
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)

        # set up samples
        self.input_mode = cfg.DATALOADER.INPUT_MODE
        self.include_incomplete = cfg.DATALOADER.INCLUDE_INCOMPLETE
        self.samples = []
        for aoi_id in self.aoi_ids:
            for sample in self.metadata[aoi_id]:
                if sample['buildings'] and not sample['masked']:
                    s1_available, s2_available = sample['s1'], sample['s2']
                    if self.include_incomplete:
                        if s1_available:
                            self.samples.append(sample)
                    else:
                        if self.input_mode == 's1' and s1_available:
                            self.samples.append(sample)
                        if self.input_mode == 's2' and s2_available:
                            self.samples.append(sample)
                        if self.input_mode == 's1s2' and s1_available and s2_available:
                            self.samples.append(sample)

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.samples = manager.list(self.samples)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        aoi_id, year, month, s1, s2 = sample['aoi_id'], sample['year'], sample['month'], sample['s1'], sample['s2']

        img_s1 = self._load_s1_img(aoi_id, year, month)
        if s2:
            img_s2 = self._load_s2_img(aoi_id, year, month)
        else:
            img_s2 = np.zeros((img_s1.shape[0], img_s1.shape[1], len(self.s2_band_indices)), dtype=np.float32)
        missing_modality = False if s1 and s2 else True

        buildings = self._load_building_label(aoi_id, year, month)

        x_s1, x_s2, y = self.transform((img_s1, img_s2, buildings))

        item = {
            'x_s1': x_s1,
            'x_s2': x_s2,
            'y': y,
            'aoi_id': aoi_id,
            'year': year,
            'month': month,
            'missing_modality': missing_modality
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
