import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from directory_management import (
    create_output_dirs_for_tiles,
    find_raw_data_dir,
    get_img_dirs,
    make_dir_if_not_exists,
    remove_temp_dirs,
)
from file_manipulation import copy_best_z_planes_to_channel_dirs
from image_stitching import stitch_images
from modify_pipeline_config import modify_pipeline_config
from slicer.slicer_runner import split_channels_into_tiles


def print_img_dirs(img_dirs: List[Path]):
    print("Image directories:")
    for dir_path in img_dirs:
        print(str(dir_path))


def load_pipeline_config(pipeline_config_path: Path) -> dict:
    with open(pipeline_config_path, "r") as s:
        submission = json.load(s)

    return submission


def convert_tiling_mode(tiling_mode: str):
    if "snake" in tiling_mode.lower():
        new_tiling_mode = "snake"
    elif "grid" in tiling_mode.lower():
        new_tiling_mode = "grid"
    else:
        raise ValueError("Unknown tiling mode: " + tiling_mode)

    return new_tiling_mode


def get_values_from_pipeline_config(pipeline_config: dict) -> dict:
    info_for_bigstitcher = dict(
        dataset_dir=Path(pipeline_config["raw_data_location"]),
        num_cycles=pipeline_config["num_cycles"],
        num_channels=len(pipeline_config["channel_names"]) // pipeline_config["num_cycles"],
        num_tiles_x=pipeline_config["region_width"],
        num_tiles_y=pipeline_config["region_height"],
        num_tiles=pipeline_config["region_width"] * pipeline_config["region_height"],
        tile_width=pipeline_config["tile_width"],
        tile_height=pipeline_config["tile_height"],
        overlap_x=pipeline_config["tile_overlap_x"],
        overlap_y=pipeline_config["tile_overlap_y"],
        overlap_z=1,  # does not matter because we have only one z-plane
        pixel_distance_x=pipeline_config["lateral_resolution"],
        pixel_distance_y=pipeline_config["lateral_resolution"],
        pixel_distance_z=pipeline_config["axial_resolution"],
        reference_channel=pipeline_config["channel_names"].index(pipeline_config["nuclei_channel"])
        + 1,
        tiling_mode=convert_tiling_mode(pipeline_config["tiling_mode"]),
        num_z_planes=pipeline_config["num_z_planes"],
        channel_names=pipeline_config["channel_names"],
    )
    return info_for_bigstitcher


def save_modified_pipeline_config(pipeline_config: dict, out_dir: Path):
    out_file_path = out_dir.joinpath("pipelineConfig.json")
    with open(out_file_path, "w") as s:
        json.dump(pipeline_config, s, indent=4)


def main(data_dir: Path, pipeline_config_path: Path):
    start = datetime.now()
    print("\nStarted", start)

    pipeline_config = load_pipeline_config(pipeline_config_path)
    dataset_meta = get_values_from_pipeline_config(pipeline_config)
    dataset_dir = find_raw_data_dir(data_dir)
    img_dirs = get_img_dirs(dataset_dir)
    print_img_dirs(img_dirs)

    best_focus_dir = Path("/output/best_focus")
    out_dir = Path("/output/processed_images")
    pipeline_conf_dir = Path("/output/pipeline_conf/")

    make_dir_if_not_exists(best_focus_dir)
    make_dir_if_not_exists(out_dir)
    make_dir_if_not_exists(pipeline_conf_dir)
    print("\nSelecting best z-planes")

    channel_dirs = copy_best_z_planes_to_channel_dirs(img_dirs, best_focus_dir, dataset_meta)

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

    remove_temp_dirs(best_focus_dir, stitched_channel_dirs)

    print("\nTime elapsed", datetime.now() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with dataset directory")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )

    args = parser.parse_args()

    main(args.data_dir, args.pipeline_config_path)
