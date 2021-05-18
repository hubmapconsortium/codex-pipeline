import os
import shutil
from math import ceil
from pathlib import Path
from typing import List


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_img_dirs(dataset_dir: Path) -> List[Path]:
    dataset_dir = dataset_dir.absolute()
    img_dir_names = next(os.walk(dataset_dir))[1]
    img_dir_paths = [dataset_dir.joinpath(dir_name).absolute() for dir_name in img_dir_names]
    return img_dir_paths


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


def create_output_dirs_for_tiles(
    stitched_channel_dirs: dict, out_dir: Path, dir_naming_template: str
):
    new_tiles_dirs = dict()
    for cycle in stitched_channel_dirs:
        new_tiles_dirs[cycle] = {}
        for region in stitched_channel_dirs[cycle]:
            new_tiles_dir_name = dir_naming_template.format(cycle=cycle, region=region)
            new_tiles_dir_path = out_dir.joinpath(new_tiles_dir_name)
            make_dir_if_not_exists(new_tiles_dir_path)
            new_tiles_dirs[cycle][region] = new_tiles_dir_path

    return new_tiles_dirs


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
    print("\nChecking if BigStitcher produced image:")
    checked_str = []
    checked_bool = []
    for cycle in stitched_channel_dirs:
        for region in stitched_channel_dirs[cycle]:
            for channel, dir_path in stitched_channel_dirs[cycle][region].items():
                if check_if_images_in_dir(dir_path):
                    checked_str.append(str(dir_path) + " passed")
                    checked_bool.append(True)
                else:
                    checked_str.append(str(dir_path) + " no image in dir")
                    checked_bool.append(False)

    print("\n".join(checked_str))

    if sum(checked_bool) < len(checked_bool):
        raise ValueError(
            "Probably there was an error while running BigStitcher. "
            + "There is no image in one or more directories."
        )
