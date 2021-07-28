from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import dask
import numpy as np
import tifffile as tif


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def project_stack(path_list: List[Path]):
    path_strs = [str(path) for path in path_list]
    stack = np.stack(list(map(tif.imread, path_strs)), axis=0)
    stack_dt = stack.dtype
    stack_mean = np.round(np.mean(stack, axis=0)).astype(stack_dt)
    return stack_mean


def process_images(src, dst):
    """ Read, take average of several z-planes, write"""
    img = project_stack(src)
    tif.imwrite(str(dst), img)


def process_images_parallelized(best_z_plane_paths: List[tuple]):
    task = []
    for src, dst in best_z_plane_paths:
        task.append(dask.delayed(process_images)(src, dst))
        # shutil.copy(src[0], dst)
    dask.compute(*task, scheduler="processes")


def process_z_planes_and_save_to_out_dirs(
    best_z_out_dirs: Dict[int, Dict[int, Path]],
    best_z_plane_paths: Dict[int, Dict[int, Dict[int, Dict[int, List[Tuple[List[Path], Path]]]]]],
):
    for cycle in best_z_out_dirs:
        for region, dir_path in best_z_out_dirs[cycle].items():
            make_dir_if_not_exists(dir_path)

    for cycle in best_z_plane_paths:
        for region in best_z_plane_paths[cycle]:
            for channel in best_z_plane_paths[cycle][region]:
                for tile, paths in best_z_plane_paths[cycle][region][channel].items():
                    process_images_parallelized(paths)
