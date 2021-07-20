import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import dask

sys.path.append("/opt/")
from directory_management import (
    create_output_dirs_for_tiles,
    get_img_dirs,
    make_dir_if_not_exists,
    remove_temp_dirs,
)
from image_stitching import stitch_images
from modify_pipeline_config import modify_pipeline_config
from slicer.slicer_runner import split_channels_into_tiles

from pipeline_utils.dataset_listing import (
    create_listing_for_each_cycle_region,
    get_img_dirs,
)
from pipeline_utils.pipeline_config_reader import load_dataset_info


def print_img_dirs(img_dirs: List[Path]):
    print("Image directories:")
    for dir_path in img_dirs:
        print(str(dir_path))


def load_pipeline_config(pipeline_config_path: Path) -> dict:
    with open(pipeline_config_path, "r") as s:
        submission = json.load(s)

    return submission


def get_file_listing(data_dir: Path):
    img_dirs = get_img_dirs(data_dir)
    listing = create_listing_for_each_cycle_region(img_dirs)
    return listing


def save_modified_pipeline_config(pipeline_config: dict, out_dir: Path):
    out_file_path = out_dir.joinpath("pipelineConfig.json")
    with open(out_file_path, "w") as s:
        json.dump(pipeline_config, s, indent=4)


def copy_to_channel_dirs(listing, base_channel_dir: Path) -> Dict[int, Dict[int, Dict[int, Path]]]:
    new_dir_name_template = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}"
    dst_name_template = "{tile:05d}.tif"
    channel_dirs = dict()
    for cycle in listing:
        channel_dirs[cycle] = dict()
        for region in listing[cycle]:
            channel_dirs[cycle][region] = dict()
            for channel in listing[cycle][region]:
                dir_name = new_dir_name_template.format(cyc=cycle, reg=region, ch=channel)
                dir_path = base_channel_dir / dir_name
                make_dir_if_not_exists(dir_path)
                channel_dirs[cycle][region][channel] = dir_path
                for tile in listing[cycle][region][channel]:
                    for zplane, src in listing[cycle][region][channel][tile].items():
                        dst_name = dst_name_template.format(tile=tile)
                        dst = dir_path / dst_name
                        shutil.copy(src, dst)
    return channel_dirs


def main(data_dir: Path, pipeline_config_path: Path):
    start = datetime.now()
    print("\nStarted", start)

    dataset_meta = load_dataset_info(pipeline_config_path)

    out_dir = Path("/output/processed_images")
    pipeline_conf_dir = Path("/output/pipeline_conf/")
    base_channel_dir = Path("/output/channel_dirs")

    make_dir_if_not_exists(out_dir)
    make_dir_if_not_exists(pipeline_conf_dir)
    make_dir_if_not_exists(base_channel_dir)

    dask.config.set({"num_workers": 5, "scheduler": "processes"})

    listing = get_file_listing(data_dir)
    channel_dirs = copy_to_channel_dirs(listing, base_channel_dir)
    stitched_channel_dirs, stitched_img_shape = stitch_images(channel_dirs, dataset_meta, out_dir)

    tile_output_dir_naming_template = "Cyc{cycle:d}_reg{region:d}"
    dirs_for_new_tiles_per_cycle_region = create_output_dirs_for_tiles(
        stitched_channel_dirs, out_dir, tile_output_dir_naming_template
    )

    print("\nSplitting channels into tiles")
    tile_size = 1000
    overlap = 100
    tile_shape = (tile_size, tile_size)
    split_channels_into_tiles(
        stitched_channel_dirs, dirs_for_new_tiles_per_cycle_region, tile_size, overlap
    )
    modified_experiment = modify_pipeline_config(
        pipeline_config_path, tile_shape, overlap, stitched_img_shape
    )

    save_modified_pipeline_config(modified_experiment, pipeline_conf_dir)

    remove_temp_dirs(stitched_channel_dirs)

    print("\nTime elapsed", datetime.now() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with image directories")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )

    args = parser.parse_args()

    main(args.data_dir, args.pipeline_config_path)
