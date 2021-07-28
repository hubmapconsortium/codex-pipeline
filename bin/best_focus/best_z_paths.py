import sys
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.append("/opt/")
from best_z_identification import get_best_z_plane_ids_per_tile

from pipeline_utils.dataset_listing import (
    create_listing_for_each_cycle_region,
    extract_digits_from_string,
)


def _change_image_file_name(original_name: str) -> str:
    """Output tiles will have names 1_00001_Z001_CH1.tif, 1_00002_Z001_CH1.tif ..."""
    digits = extract_digits_from_string(original_name)
    region = digits[0]
    tile = digits[1]
    zplane = 1
    channel = digits[3]
    return "{reg:d}_{tile:05d}_Z{z:03d}_CH{ch:d}.tif".format(
        reg=region, tile=tile, z=zplane, ch=channel
    )


def _get_reference_channel_paths(
    listing_per_cycle: dict, num_channels_per_cycle: int, reference_channel_id: int
) -> Dict[int, Path]:
    ref_cycle_id = ceil(reference_channel_id / num_channels_per_cycle) - 1
    ref_cycle = sorted(listing_per_cycle.keys())[ref_cycle_id]
    ref_cycle_ref_channel_id = reference_channel_id - ref_cycle_id * num_channels_per_cycle

    reference_channel_tile_paths = dict()
    for region in listing_per_cycle[ref_cycle]:
        reference_channel_tile_paths.update({region: {}})
        this_channel_tile_paths = listing_per_cycle[ref_cycle][region][ref_cycle_ref_channel_id]
        reference_channel_tile_paths[region] = this_channel_tile_paths
    return reference_channel_tile_paths


def _create_dirs_for_each_cycle_region(
    listing_per_cycle: dict, out_dir: Path
) -> Dict[int, Dict[int, Path]]:
    naming_template = "Cyc{cyc:03d}_reg{reg:03d}"
    cyc_reg_dirs = dict()
    for cycle in listing_per_cycle:
        cyc_reg_dirs[cycle] = dict()
        for region in listing_per_cycle[cycle]:
            dir_name = naming_template.format(cyc=cycle, reg=region)
            cyc_reg_dirs[cycle][region] = out_dir / dir_name
    return cyc_reg_dirs


def _find_best_z_planes_per_region_tile(
    reference_channel_tile_paths: dict, max_z: int, x_ntiles: int, y_ntiles: int
) -> Dict[int, Dict[int, List[int]]]:
    best_z_plane_per_region = dict()

    for region in reference_channel_tile_paths:
        best_z_plane_per_region[region] = get_best_z_plane_ids_per_tile(
            reference_channel_tile_paths[region], x_ntiles, y_ntiles, max_z
        )  # output {region: {tile: [ids] }}
    return best_z_plane_per_region


def _map_best_z_planes_in_channel_to_output_plane(
    channel_paths: dict, out_dir: Path, best_z_plane_per_tile: dict
) -> List[Tuple[List[Path], Path]]:
    best_z_plane_paths = list()
    for tile in channel_paths:
        this_tile_paths = channel_paths[tile]
        best_focal_plane_ids = best_z_plane_per_tile[tile]  # list of ids

        best_z_input_paths = []
        for _id in best_focal_plane_ids:
            best_z_input_paths.append(this_tile_paths[_id])

        best_z_file_name = best_z_input_paths[0].name
        output_combined_name = _change_image_file_name(best_z_file_name)
        output_combined_path = Path(out_dir).joinpath(output_combined_name)

        best_z_plane_paths.append((best_z_input_paths, output_combined_path))

    return best_z_plane_paths


def _select_best_z_plane_paths(
    listing: Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]],
    out_dirs: Dict[int, Dict[int, Path]],
    best_z_plane_per_region: Dict[int, Dict[int, List[int]]],
) -> Dict[int, Dict[int, Dict[int, Dict[int, List[Tuple[List[Path], Path]]]]]]:
    """Creates a map of several raw planes that will be processed into one image"""
    best_z_plane_paths = dict()
    for cycle in listing:
        best_z_plane_paths[cycle] = dict()
        for region in listing[cycle]:
            best_z_plane_paths[cycle][region] = dict()
            this_cyc_reg_out_dir = out_dirs[cycle][region]
            this_region_best_z_planes = best_z_plane_per_region[region]
            for channel in listing[cycle][region]:
                best_z_plane_paths[cycle][region][channel] = dict()
                for tile, zplane_dict in listing[cycle][region][channel].items():

                    this_tile_best_z_ids = this_region_best_z_planes[tile]
                    this_tile_best_z_src_paths = []
                    for _id in this_tile_best_z_ids:
                        this_tile_best_z_src_paths.append(zplane_dict[_id])

                    best_z_file_name = this_tile_best_z_src_paths[0].name
                    this_tile_best_z_dst_combined_name = _change_image_file_name(best_z_file_name)
                    this_tile_best_z_dst_combined_path = (
                        this_cyc_reg_out_dir / this_tile_best_z_dst_combined_name
                    )

                    if tile in best_z_plane_paths[cycle][region][channel]:
                        best_z_plane_paths[cycle][region][channel][tile].append(
                            (this_tile_best_z_src_paths, this_tile_best_z_dst_combined_path)
                        )
                    else:
                        best_z_plane_paths[cycle][region][channel][tile] = [
                            (this_tile_best_z_src_paths, this_tile_best_z_dst_combined_path)
                        ]
    return best_z_plane_paths


def get_best_z_dirs_and_paths(
    img_dirs: List[Path],
    out_dir: Path,
    num_channels_per_cycle: int,
    max_z: int,
    x_ntiles: int,
    y_ntiles: int,
    reference_channel_id: int,
) -> Tuple[
    Dict[int, Dict[int, Path]],
    Dict[int, Dict[int, Dict[int, Dict[int, List[Tuple[List[Path], Path]]]]]],
]:
    listing_per_cycle = create_listing_for_each_cycle_region(img_dirs)
    reference_channel_tile_paths = _get_reference_channel_paths(
        listing_per_cycle, num_channels_per_cycle, reference_channel_id
    )
    best_z_dirs = _create_dirs_for_each_cycle_region(listing_per_cycle, out_dir)
    best_z_plane_per_region = _find_best_z_planes_per_region_tile(
        reference_channel_tile_paths, max_z, x_ntiles, y_ntiles
    )
    best_z_plane_paths = _select_best_z_plane_paths(
        listing_per_cycle, best_z_dirs, best_z_plane_per_region
    )
    return best_z_dirs, best_z_plane_paths


def find_best_z_paths_and_dirs(
    dataset_info: Dict[str, Any], img_dirs: List[Path], out_dir: Path
) -> Tuple[
    Dict[int, Dict[int, Path]],
    Dict[int, Dict[int, Dict[int, Dict[int, List[Tuple[List[Path], Path]]]]]],
]:
    nzplanes = dataset_info["num_z_planes"]
    x_ntiles = dataset_info["num_tiles_x"]
    y_ntiles = dataset_info["num_tiles_y"]
    reference_channel_id = dataset_info["reference_channel"]
    num_channels_per_cycle = dataset_info["num_channels"]

    best_z_channel_dirs, best_z_plane_paths = get_best_z_dirs_and_paths(
        img_dirs,
        out_dir,
        num_channels_per_cycle,
        nzplanes,
        x_ntiles,
        y_ntiles,
        reference_channel_id,
    )
    return best_z_channel_dirs, best_z_plane_paths
