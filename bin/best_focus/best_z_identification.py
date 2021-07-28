from pathlib import Path
from typing import Dict, List

import cv2 as cv
import dask
import numpy as np
import tifffile as tif
from scipy.ndimage import gaussian_filter

Image = np.ndarray


def _laplacian_variance(img: Image) -> float:
    """
    DOI:10.1016/j.patcog.2012.11.011
    Analysis of focus measure operators for shape-from-focus
    """
    return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))


def _find_best_z_plane_id(img_list: List[Image]) -> int:
    lap_vars_per_z_plane = []
    for img in img_list:
        lap_vars_per_z_plane.append(_laplacian_variance(img))
    max_var = max(lap_vars_per_z_plane)
    max_var_id = lap_vars_per_z_plane.index(max_var)
    return max_var_id


def _load_images(path_list: List[Path]) -> List[Image]:
    img_list = []
    for path in path_list:
        img_list.append(tif.imread(str(path)))
    return img_list


def get_best_z_plane_id(path_list: List[Path]) -> int:
    img_list = _load_images(path_list)
    return _find_best_z_plane_id(img_list) + 1


def get_best_z_plane_id_parallelized(plane_paths_per_tile: dict) -> List[int]:
    task = []
    for tile, plane_paths in plane_paths_per_tile.items():
        plane_path_list = list(plane_paths.values())
        task.append(dask.delayed(get_best_z_plane_id)(plane_path_list))
    best_z_plane_id_list = dask.compute(*task)
    best_z_plane_id_list = list(best_z_plane_id_list)
    return best_z_plane_id_list


def smoothing_z_ids(arr: np.ndarray):
    smoothed_ids_float = gaussian_filter(arr.astype(np.float32), 1, mode="reflect")
    smoothed_ids = np.round(smoothed_ids_float, 0).astype(np.uint32)
    return smoothed_ids


def best_z_correction(best_z_plane_id_list: List[int], x_ntiles: int, y_ntiles: int) -> np.ndarray:
    best_z_per_tile_arr = np.array(best_z_plane_id_list, dtype=np.int32).reshape(
        y_ntiles, x_ntiles
    )
    print("Best z-plane per tile")
    print("Original values\n", best_z_per_tile_arr)
    smoothed_best_z_per_tile_arr = smoothing_z_ids(best_z_per_tile_arr)
    print("Corrected values\n", smoothed_best_z_per_tile_arr)
    result = smoothed_best_z_per_tile_arr.ravel().tolist()

    return result


def pick_z_planes_below_and_above(best_z: int, max_z: int, above: int, below: int) -> List[int]:
    range_end = best_z + above
    if range_end > max_z:
        range_end = max_z

    range_start = best_z - below
    if range_start < 1:
        range_start = 1

    if max_z == 1:
        return [best_z]
    elif best_z == max_z:
        below_planes = list(range(range_start, best_z))
        above_planes = []
    elif best_z == 1:
        below_planes = []
        above_planes = list(range(best_z + 1, range_end + 1))
    else:
        below_planes = list(range(range_start, best_z))
        above_planes = list(range(best_z + 1, range_end + 1))
    return below_planes + [best_z] + above_planes


def get_best_z_plane_ids_per_tile(
    plane_paths_per_tile: dict, x_ntiles: int, y_ntiles: int, max_z: int
) -> Dict[int, List[int]]:
    best_z_plane_id_list = get_best_z_plane_id_parallelized(plane_paths_per_tile)
    corrected_best_z_plane_id_list = best_z_correction(best_z_plane_id_list, x_ntiles, y_ntiles)

    best_z_plane_per_tile = dict()
    for i, tile in enumerate(plane_paths_per_tile.keys()):
        best_z_plane_per_tile[tile] = pick_z_planes_below_and_above(
            corrected_best_z_plane_id_list[i], max_z, 1, 1
        )
    return best_z_plane_per_tile
