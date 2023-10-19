import platform
import shutil
import subprocess
from pathlib import Path
from typing import List

import dask
import tifffile as tif
from directory_management import (
    check_stitched_dirs,
    create_dirs_for_stitched_channels,
    get_ref_channel_dir_per_region,
)
from generate_bigstitcher_macro import BigStitcherMacro, FuseMacro


def get_image_path_in_dir(dir_path: Path) -> Path:
    allowed_extensions = (".tif", ".tiff")
    listing = list(dir_path.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    return img_listing[0]


def generate_bigstitcher_macro_for_reference_channel(
    reference_channel_dir: Path, out_dir: Path, dataset_info: dict, region: int
) -> Path:
    tile_shape = (
        dataset_info["tile_height"] + dataset_info["overlap_y"],
        dataset_info["tile_width"] + dataset_info["overlap_x"],
    )

    macro = BigStitcherMacro()
    macro.img_dir = reference_channel_dir
    macro.out_dir = out_dir
    macro.pattern = "{xxxxx}.tif"
    macro.num_tiles = dataset_info["num_tiles"]
    macro.num_tiles_x = dataset_info["num_tiles_x"]
    macro.num_tiles_y = dataset_info["num_tiles_y"]
    macro.tile_shape = tile_shape
    macro.overlap_x = dataset_info["overlap_x"]
    macro.overlap_y = dataset_info["overlap_y"]
    macro.overlap_z = dataset_info["overlap_z"]
    macro.pixel_distance_x = dataset_info["pixel_distance_x"]
    macro.pixel_distance_y = dataset_info["pixel_distance_y"]
    macro.pixel_distance_z = dataset_info["pixel_distance_z"]
    macro.tiling_mode = dataset_info["tiling_mode"]
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
    else:
        raise ValueError(f"unsupported platform: {platform.system()}")

    command = imagej_name + " --headless --console -macro " + str(bigstitcher_macro_path)
    print("Started running BigStitcher for", str(bigstitcher_macro_path))
    res = subprocess.run(
        command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if res.returncode == 0:
        print("Finished", str(bigstitcher_macro_path))
    else:
        raise Exception(
            "There was an error while running the BigStitcher for "
            + str(bigstitcher_macro_path)
            + "\n"
            + res.stderr.decode("utf-8")
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
                macro_path = dir_path.joinpath("fuse_macro.ijm")
                task.append(dask.delayed(run_bigstitcher)(macro_path))

    dask.compute(*task, scheduler="processes")


def get_stitched_image_shape(ref_channel_stitched_dir_per_region):
    for region, dir_path in ref_channel_stitched_dir_per_region.items():
        stitched_image_path = get_image_path_in_dir(dir_path)
        break
    with tif.TiffFile(stitched_image_path) as TF:
        stitched_image_shape = TF.series[0].shape

    return stitched_image_shape


def stitch_images(channel_dirs, dataset_meta, out_dir):
    ref_channel_id = int(dataset_meta["reference_channel"])
    num_channels_per_cycle = dataset_meta["num_channels"]

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

    return stitched_channel_dirs, stitched_img_shape
