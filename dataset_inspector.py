import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from utils import geofiles, parsers


def plot_satellite_data(dataset_path: str, output_path: str):
    metadata_file = Path(dataset_path) / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    aoi_ids = metadata.keys()
    for aoi_id in aoi_ids:
        timestamps = metadata[aoi_id]
        for timestamp in timestamps:
            if timestamp['s1']:
                year, month = timestamp['year'], timestamp['month']
                fig, axs = plt.subplots(3, 1, figsize=(5, 15))

                s1_file = Path(dataset_path) / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
                s1, *_ = geofiles.read_tif(s1_file)
                axs[0].imshow(s1[:, :, 0], cmap='gray', vmin=0, vmax=1)

                s2_file = Path(dataset_path) / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'

                if s2_file.exists():
                    s2, *_ = geofiles.read_tif(s2_file)
                    axs[1].imshow(np.clip(s2[:, :, [2, 1, 0]] / 0.4, 0, 1), vmin=0, vmax=1)

                buildings_file = Path(dataset_path) / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
                if buildings_file.exists():
                    buildings, *_ = geofiles.read_tif(buildings_file)
                    axs[2].imshow(buildings.squeeze() > 0, cmap='gray', vmin=0, vmax=1)

                for _, ax in np.ndenumerate(axs):
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.tight_layout()
                out_file = Path(output_path) / 'plots' / 'dataset' / f'{aoi_id}_{year}_{month:02d}.png'
                plt.savefig(out_file, dpi=300, bbox_inches='tight')
                plt.close(fig)


def plot_timeseries(dataset_path: str, output_path: str):
    metadata_file = Path(dataset_path) / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    aoi_ids = metadata.keys()
    for aoi_id in aoi_ids:
        timestamps = metadata[aoi_id]
        timestamps = [t for t in timestamps if t['s1'] and t['buildings'] and not t['masked']]
        m, n = 3, len(timestamps)
        fig, axs = plt.subplots(m, n, figsize=(n * 3, m * 3))
        for i, timestamp in enumerate(timestamps):
            year, month = timestamp['year'], timestamp['month']
            # print(f'{aoi_id} {year}-{month:02d}')
            s1_file = Path(dataset_path) / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
            s1, *_ = geofiles.read_tif(s1_file)
            axs[0, i].imshow(s1[:, :, 0], cmap='gray', vmin=0, vmax=1)

            s2_file = Path(dataset_path) / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
            if s2_file.exists():
                s2, *_ = geofiles.read_tif(s2_file)
                axs[1, i].imshow(np.clip(s2[:, :, [2, 1, 0]] / 0.4, 0, 1), vmin=0, vmax=1)

            buildings_file = Path(dataset_path) / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
            buildings, *_ = geofiles.read_tif(buildings_file)
            axs[2, i].imshow(buildings.squeeze(), cmap='gray', vmin=0, vmax=1)

            for _, ax in np.ndenumerate(axs):
                ax.set_xticks([])
                ax.set_yticks([])

        out_file = Path(output_path) / 'plots' / 'dataset' / f'timeseries_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    args = parsers.dataset_argument_parser().parse_known_args()[0]
    plot_satellite_data(args.dataset_dir, args.output_dir)
