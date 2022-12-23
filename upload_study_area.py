import ee
from pathlib import Path
from utils import experiment_manager, spacenet7_helpers, geofiles, parsers


def get_centroid(aoi_id: str, spacenet7_path: str, dataset: str) -> ee.Geometry:
    folder = Path(spacenet7_path) / dataset / aoi_id / 'images_masked'
    files = [file for file in folder.glob('**/*') if file.is_file()]
    _, transform, crs = geofiles.read_tif(files[0])
    _, _, c, _, _, f, *_ = transform
    return ee.Geometry.Point(coords=[c, f], proj=str(crs)).transform()


def upload_study_area(spacenet7_path: str):
    cfg = experiment_manager.load_cfg('base')
    aoi_ids = spacenet7_helpers.get_all_aoi_ids(spacenet7_path, 'train')
    features = []
    for aoi_id in aoi_ids:
        centroid = get_centroid(aoi_id, spacenet7_path, 'train')
        if aoi_id in cfg.DATASET.TRAINING_IDS:
            split = 'training'
        else:
            split = 'test'
        features.append(ee.Feature(centroid, {'aoi_id': aoi_id, 'split': split}))

    fc = ee.FeatureCollection(features)
    dl_task = ee.batch.Export.table.toDrive(
        collection=fc,
        description='studyAreaMissingModality',
        folder='missing_modality',
        fileNamePrefix='aoi_ids',
        fileFormat='GeoJSON'
    )
    dl_task.start()


if __name__ == '__main__':
    ee.Initialize()
    args = parsers.study_area_upload_argument_parser().parse_known_args()[0]
    upload_study_area(args.spacenet7_dir)