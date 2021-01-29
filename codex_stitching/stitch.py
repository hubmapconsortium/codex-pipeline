import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import List

import dask
import tifffile as tif
from file_manipulation import copy_best_z_planes_to_channel_dirs
from generate_bigstitcher_macro import BigStitcherMacro, FuseMacro
from modify_pipeline_config import modify_pipeline_config
from slicer.slicer_runner import get_image_path_in_dir, split_channels_into_tiles


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


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


def generate_bigstitcher_macro_for_reference_channel(
    reference_channel_dir: Path, out_dir: Path, info_for_bigstitcher: dict, region: int
) -> Path:
    tile_shape = (
        info_for_bigstitcher["tile_height"] + info_for_bigstitcher["overlap_y"],
        info_for_bigstitcher["tile_width"] + info_for_bigstitcher["overlap_x"],
    )

    macro = BigStitcherMacro()
    macro.img_dir = reference_channel_dir
    macro.out_dir = out_dir
    macro.pattern = "1_{xxxxx}_Z001.tif"
    macro.num_tiles = info_for_bigstitcher["num_tiles"]
    macro.num_tiles_x = info_for_bigstitcher["num_tiles_x"]
    macro.num_tiles_y = info_for_bigstitcher["num_tiles_y"]
    macro.tile_shape = tile_shape
    macro.overlap_x = info_for_bigstitcher["overlap_x"]
    macro.overlap_y = info_for_bigstitcher["overlap_y"]
    macro.overlap_z = info_for_bigstitcher["overlap_z"]
    macro.pixel_distance_x = info_for_bigstitcher["pixel_distance_x"]
    macro.pixel_distance_y = info_for_bigstitcher["pixel_distance_y"]
    macro.pixel_distance_z = info_for_bigstitcher["pixel_distance_z"]
    macro.tiling_mode = info_for_bigstitcher["tiling_mode"]
    macro.region = region
    macro_path = macro.generate()

    return macro_path


def run_bigstitcher(bigstitcher_macro_path: Path):
    # It is expected that ImageJ is added to system PATH

    if platform.system() == "Windows":
        imagej_name = "ImageJ-win64"
    elif platform.system() == "Linux":
        imagej_name = "ImageJ-linux64"
    elif platform.system() == "Darwin":
        imagej_name = "ImageJ-macosx"

    command = imagej_name + " --headless --console -macro " + str(bigstitcher_macro_path)
    print("Started running BigStitcher")
    res = subprocess.run(
        command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if res.returncode == 0:
        print("Successfully finished")
    else:
        raise Exception(
            "There was an error while running the BigStitcher: \n" + res.stderr.decode("utf-8")
        )


def run_bigstitcher_for_ref_channel_per_region(
    ref_channel_dir_per_region: dict,
    ref_channel_stitched_dir_per_region: dict,
    info_for_bigstitcher: dict,
):
    for region, dir_path in ref_channel_dir_per_region.items():
        ref_channel_dir = dir_path
        ref_channel_stitched_dir = ref_channel_stitched_dir_per_region[region]
        bigstitcher_macro_path = generate_bigstitcher_macro_for_reference_channel(
            ref_channel_dir, ref_channel_stitched_dir, info_for_bigstitcher, region
        )
        run_bigstitcher(bigstitcher_macro_path)


def copy_dataset_xml_to_channel_dirs(ref_channel_dir: Path, other_channel_dirs: List[Path]):
    dataset_xml_path = ref_channel_dir.joinpath("dataset.xml")
    for dir_path in other_channel_dirs:
        dst_path = dir_path.joinpath("dataset.xml")
        try:
            shutil.copy(dataset_xml_path, dst_path)
        except shutil.SameFileError:
            continue


def copy_fuse_macro_to_channel_dirs(channel_dirs: List[Path], channel_stitched_dirs: List[Path]):
    macro = FuseMacro()
    for i, dir_path in enumerate(channel_dirs):
        macro.img_dir = dir_path
        macro.xml_file_name = "dataset.xml"
        macro.out_dir = channel_stitched_dirs[i]
        macro.generate()


def copy_bigsticher_files_to_dirs(
    channel_dirs: dict, stitched_channel_dirs: dict, ref_channel_dir_per_region: dict
):
    for cycle in channel_dirs:
        for region in channel_dirs[cycle]:
            this_region_ref_channel_dir = ref_channel_dir_per_region[region]
            channel_dir_list = list(channel_dirs[cycle][region].values())
            channel_stitched_dir_list = list(stitched_channel_dirs[cycle][region].values())

            copy_dataset_xml_to_channel_dirs(this_region_ref_channel_dir, channel_dir_list)
            copy_fuse_macro_to_channel_dirs(channel_dir_list, channel_stitched_dir_list)


def run_stitching_for_all_channels(channel_dirs: dict):
    task = []
    for cycle in channel_dirs:
        for region in channel_dirs[cycle]:
            for channel, dir_path in channel_dirs[cycle][region].items():
                macro_path = dir_path.joinpath("fuse_only_macro.ijm")
                task.append(dask.delayed(run_bigstitcher)(macro_path))

    dask.compute(*task, scheduler="processes")


def create_dirs_for_stitched_channels(channel_dirs: dict, out_dir: Path):
    stitched_channel_dirs = dict()
    for cycle in channel_dirs:
        stitched_channel_dirs[cycle] = {}
        for region in channel_dirs[cycle]:
            stitched_channel_dirs[cycle][region] = {}
            for channel, dir_path in channel_dirs[cycle][region].items():
                dirname = Path(dir_path).name
                stitched_dir_path = out_dir.joinpath(dirname)
                stitched_channel_dirs[cycle][region][channel] = stitched_dir_path
                make_dir_if_not_exists(stitched_dir_path)

    return stitched_channel_dirs


def get_ref_channel_dir_per_region(
    channel_dirs: dict,
    stitched_channel_dirs: dict,
    num_channels_per_cycle: int,
    reference_channel_id: int,
):
    ref_cycle_id = ceil(reference_channel_id / num_channels_per_cycle) - 1
    ref_cycle = sorted(channel_dirs.keys())[ref_cycle_id]
    in_cycle_ref_channel_id = reference_channel_id - ref_cycle_id * num_channels_per_cycle

    reference_channel_dir = dict()
    for region in channel_dirs[ref_cycle]:
        this_channel_dir = channel_dirs[ref_cycle][region][in_cycle_ref_channel_id]
        reference_channel_dir[region] = this_channel_dir

    stitched_ref_channel_dir = dict()
    for region in stitched_channel_dirs[ref_cycle]:
        this_channel_dir = stitched_channel_dirs[ref_cycle][region][in_cycle_ref_channel_id]
        stitched_ref_channel_dir[region] = this_channel_dir

    return reference_channel_dir, stitched_ref_channel_dir


def create_dirs_for_new_tiles_per_cycle_region(stitched_channel_dirs: dict, out_dir: Path):
    dir_naming_template = "Cyc{cycle:d}_reg{region:d}"
    new_tiles_dirs = dict()
    for cycle in stitched_channel_dirs:
        new_tiles_dirs[cycle] = {}
        for region in stitched_channel_dirs[cycle]:
            new_tiles_dir_name = dir_naming_template.format(cycle=cycle, region=region)
            new_tiles_dir_path = out_dir.joinpath(new_tiles_dir_name)
            make_dir_if_not_exists(new_tiles_dir_path)
            new_tiles_dirs[cycle][region] = new_tiles_dir_path

    return new_tiles_dirs


def get_stitched_image_shape(ref_channel_stitched_dir_per_region):
    for region, dir_path in ref_channel_stitched_dir_per_region.items():
        stitched_image_path = get_image_path_in_dir(dir_path)
        break
    with tif.TiffFile(stitched_image_path) as TF:
        stitched_image_shape = TF.series[0].shape

    return stitched_image_shape


def save_modified_pipeline_config(pipeline_config: dict, out_dir: Path):
    out_file_path = out_dir.joinpath("pipelineConfig.json")
    with open(out_file_path, "w") as s:
        json.dump(pipeline_config, s, indent=4)


def get_img_dirs(dataset_dir: Path) -> List[Path]:
    img_dir_names = next(os.walk(dataset_dir))[1]
    img_dir_paths = [dataset_dir.joinpath(dir_name) for dir_name in img_dir_names]
    return img_dir_paths


def remove_temp_dirs(best_focus_dir: Path, stitched_channel_dirs: dict):
    shutil.rmtree(str(best_focus_dir))

    for cycle in stitched_channel_dirs:
        for region in stitched_channel_dirs[cycle]:
            for channel, dir_path in stitched_channel_dirs[cycle][region].items():
                shutil.rmtree(str(dir_path))


def check_if_images_in_dir(dir_path: Path):
    allowed_extensions = (".tif", ".tiff")
    listing = list(dir_path.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    if img_listing:
        return True
    else:
        return False


def check_stitched_dirs(stitched_channel_dirs: dict):
    for cycle in stitched_channel_dirs:
        for region in stitched_channel_dirs[cycle]:
            for channel, dir_path in stitched_channel_dirs[cycle][region].items():
                print(dir_path, check_if_images_in_dir(dir_path))


def find_raw_data_dir(directory: Path) -> Path:
    NONRAW_DIRECTORY_NAME_PIECES = ["processed", "drv", "metadata"]

    raw_data_dir_possibilities = []

    for child in directory.iterdir():
        if not child.is_dir():
            continue
        if not any(piece in child.name for piece in NONRAW_DIRECTORY_NAME_PIECES):
            raw_data_dir_possibilities.append(child)

    if len(raw_data_dir_possibilities) > 1:
        message_pieces = ["Found multiple raw data directory possibilities:"]
        message_pieces.extend("\t" + str(path) for path in raw_data_dir_possibilities)
        raise ValueError("\n".join(message_pieces))

    return raw_data_dir_possibilities[0]


def main(data_dir: Path, pipeline_config_path: Path):
    start = datetime.now()
    print("\nStarted", start)

    pipeline_config = load_pipeline_config(pipeline_config_path)
    dataset_meta = get_values_from_pipeline_config(pipeline_config)
    dataset_dir = find_raw_data_dir(data_dir)

    img_dirs = get_img_dirs(dataset_dir)
    print("Image directories:", [str(dir_path) for dir_path in img_dirs])

    best_focus_dir = Path("/output/best_focus")
    out_dir = Path("/output/processed_images")
    pipeline_conf_dir = Path("/output/pipeline_conf/")

    make_dir_if_not_exists(best_focus_dir)
    make_dir_if_not_exists(out_dir)
    make_dir_if_not_exists(pipeline_conf_dir)

    ref_channel_id = int(dataset_meta["reference_channel"])
    num_channels_per_cycle = dataset_meta["num_channels"]
    print("\nSelecting best z-planes")

    channel_dirs = copy_best_z_planes_to_channel_dirs(img_dirs, best_focus_dir, dataset_meta)
    stitched_channel_dirs = create_dirs_for_stitched_channels(channel_dirs, out_dir)

    ref_ch_dirs = get_ref_channel_dir_per_region(
        channel_dirs, stitched_channel_dirs, num_channels_per_cycle, ref_channel_id
    )
    ref_channel_dir_per_region, ref_channel_stitched_dir_per_region = ref_ch_dirs

    print("\nEstimating stitching parameters")
    run_bigstitcher_for_ref_channel_per_region(
        ref_channel_dir_per_region, ref_channel_stitched_dir_per_region, dataset_meta
    )

    print("\nStitching channels")
    copy_bigsticher_files_to_dirs(channel_dirs, stitched_channel_dirs, ref_channel_dir_per_region)
    run_stitching_for_all_channels(channel_dirs)
    check_stitched_dirs(stitched_channel_dirs)
    stitched_img_shape = get_stitched_image_shape(ref_channel_stitched_dir_per_region)
    new_dirs_tiles_per_cycle_region = create_dirs_for_new_tiles_per_cycle_region(
        stitched_channel_dirs, out_dir
    )

    print("\nSplitting channels into tiles")
    block_size = 1000
    overlap = 100
    block_shape = (block_size, block_size)
    split_channels_into_tiles(
        stitched_channel_dirs, new_dirs_tiles_per_cycle_region, block_size, overlap
    )
    modified_experiment = modify_pipeline_config(
        pipeline_config_path, block_shape, overlap, stitched_img_shape
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
