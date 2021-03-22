from pathlib import Path
from typing import Dict, List, Set, Union

import dask
import numpy as np
import tifffile as tif
from best_z_paths import get_output_dirs_and_paths


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def project_stack(path_list: List[Path]):
    path_strs = [str(path) for path in path_list]
    stack = np.stack(list(map(tif.imread, path_strs)), axis=0)
    stack_dt = stack.dtype
    stack_mean = np.round(np.mean(stack, axis=0)).astype(stack_dt)
    return stack_mean


def read_and_write(src, dst):
    img = project_stack(src)
    tif.imwrite(str(dst), img)


def write_to_destination(best_z_plane_paths: List[tuple]):
    task = []
    for src, dst in best_z_plane_paths:
        task.append(dask.delayed(read_and_write)(src, dst))
        # shutil.copy(src[0], dst)
    dask.compute(*task, scheduler="processes")


def get_channel_names_per_cycle(dataset_meta: dict):
    channel_names = dataset_meta["channel_names"]
    channels_per_cycle = dataset_meta["num_channels"]
    channel_ids = list(range(0, len(channel_names)))
    cycle_boundaries = channel_ids[::channels_per_cycle]
    cycle_boundaries.append(len(channel_names))

    channel_names_per_cycle = dict()
    for i in range(0, len(cycle_boundaries) - 1):
        f = cycle_boundaries[i]  # from
        t = cycle_boundaries[i + 1]  # to
        channel_names_per_cycle[i + 1] = channel_names[f:t]

    return channel_names_per_cycle


def get_tile_and_plane_info(dataset_meta: dict):
    nzplanes = dataset_meta["num_z_planes"]
    x_ntiles = dataset_meta["num_tiles_x"]
    y_ntiles = dataset_meta["num_tiles_y"]

    return nzplanes, x_ntiles, y_ntiles


def copy_best_z_planes_to_channel_dirs(img_dirs: List[Path], out_dir: Path, dataset_meta: dict):
    nzplanes, x_ntiles, y_ntiles = get_tile_and_plane_info(dataset_meta)
    tiling_mode = dataset_meta["tiling_mode"]
    reference_channel = dataset_meta["reference_channel"]
    num_channels_per_cycle = dataset_meta["num_channels"]
    channel_dirs, best_z_plane_paths = get_output_dirs_and_paths(
        img_dirs,
        out_dir,
        num_channels_per_cycle,
        nzplanes,
        x_ntiles,
        y_ntiles,
        tiling_mode,
        reference_channel,
    )

    for cycle in channel_dirs:
        for region in channel_dirs[cycle]:
            for channel, dir_path in channel_dirs[cycle][region].items():
                make_dir_if_not_exists(dir_path)

    for cycle in best_z_plane_paths:
        for region in best_z_plane_paths[cycle]:
            for channel, paths in best_z_plane_paths[cycle][region].items():
                write_to_destination(paths)

    return channel_dirs
