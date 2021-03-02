import re
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

from best_z_identification import get_best_z_plane_ids_per_tile
from image_path_arrangement import create_listing_for_each_cycle_region


def _change_image_file_name(original_name: str) -> str:
    sub_z = re.sub(r"Z\d{3}", "Z001", original_name)  # replace z plane id (Z) in all tiles to Z001
    sub_ch = re.sub(r"_CH\d+\.", ".", sub_z)  # remove channel (CH) from name

    return sub_ch


def _get_reference_channel_paths(
    listing_per_cycle: dict, num_channels_per_cycle: int, reference_channel_id: int
) -> Dict[int, Path]:
    ref_cycle_id = ceil(reference_channel_id / num_channels_per_cycle) - 1
    ref_cycle = sorted(listing_per_cycle.keys())[ref_cycle_id]
    in_cycle_ref_channel_id = reference_channel_id - ref_cycle_id * num_channels_per_cycle

    reference_channel_tile_paths = dict()
    for region in listing_per_cycle[ref_cycle]:
        reference_channel_tile_paths.update({region: {}})
        this_channel_tile_paths = listing_per_cycle[ref_cycle][region][in_cycle_ref_channel_id]
        reference_channel_tile_paths[region] = this_channel_tile_paths

    return reference_channel_tile_paths


def _create_dirs_for_each_channel(
    listing_per_cycle: dict, out_dir: Path
) -> Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]]:
    naming_template = "Cyc{cycle:03d}_Reg{region:03d}_CH{channel:03d}"

    channel_dirs = dict()
    for cycle in listing_per_cycle:
        this_cycle_listing = listing_per_cycle[cycle]
        channel_dirs.update({cycle: {}})
        for region in this_cycle_listing:
            this_region_listing = this_cycle_listing[region]
            channel_dirs[cycle].update({region: dict()})
            for channel in this_region_listing:
                channel_dir_name = naming_template.format(
                    cycle=cycle, region=region, channel=channel
                )
                this_channel_out_dir = out_dir.joinpath(channel_dir_name)
                channel_dirs[cycle][region].update({channel: this_channel_out_dir})

    return channel_dirs


def _find_best_z_planes_per_region_tile(
    reference_channel_tile_paths: dict, max_z: int, x_ntiles: int, y_ntiles: int, tiling_mode: str
) -> Dict[int, Dict[int, List[Path]]]:
    best_z_plane_per_region = dict()

    for region in reference_channel_tile_paths:
        best_z_plane_per_tile = get_best_z_plane_ids_per_tile(
            reference_channel_tile_paths[region], x_ntiles, y_ntiles, max_z, tiling_mode
        )
        best_z_plane_per_region[region] = best_z_plane_per_tile

    return best_z_plane_per_region


def _select_best_z_planes_in_this_channel(
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
    listing_per_cycle: dict, channel_dirs: dict, best_z_plane_per_region: dict
) -> Dict[int, Dict[int, Dict[int, List[Tuple[List[Path], Path]]]]]:
    best_z_plane_paths_per_cycle = dict()

    for cycle in listing_per_cycle:
        this_cycle_listing = listing_per_cycle[cycle]
        best_z_plane_paths_per_cycle.update({cycle: {}})
        for region in this_cycle_listing:
            this_region_listing = this_cycle_listing[region]
            best_z_plane_paths_per_cycle[cycle].update({region: {}})
            for channel in this_region_listing:
                this_channel_paths = this_region_listing[channel]
                this_channel_out_dir = channel_dirs[cycle][region][channel]
                this_region_best_z_planes = best_z_plane_per_region[region]
                best_z_plane_paths = _select_best_z_planes_in_this_channel(
                    this_channel_paths, this_channel_out_dir, this_region_best_z_planes
                )

                best_z_plane_paths_per_cycle[cycle][region].update({channel: best_z_plane_paths})

    return best_z_plane_paths_per_cycle


def get_output_dirs_and_paths(
    img_dirs: List[Path],
    out_dir: Path,
    num_channels_per_cycle: int,
    max_z: int,
    x_ntiles: int,
    y_ntiles: int,
    tiling_mode: str,
    reference_channel_id: int,
):
    listing_per_cycle = create_listing_for_each_cycle_region(img_dirs)
    reference_channel_tile_paths = _get_reference_channel_paths(
        listing_per_cycle, num_channels_per_cycle, reference_channel_id
    )
    channel_dirs = _create_dirs_for_each_channel(listing_per_cycle, out_dir)
    best_z_plane_per_region = _find_best_z_planes_per_region_tile(
        reference_channel_tile_paths, max_z, x_ntiles, y_ntiles, tiling_mode
    )
    best_z_plane_paths = _select_best_z_plane_paths(
        listing_per_cycle, channel_dirs, best_z_plane_per_region
    )

    return channel_dirs, best_z_plane_paths
