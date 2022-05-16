import json
import os
from pathlib import Path
from typing import List


def load_pipeline_config(pipeline_config_path: Path) -> dict:
    with open(pipeline_config_path, "r") as s:
        config = json.load(s)
    return config


def _convert_tiling_mode(tiling_mode: str):
    if "snake" in tiling_mode.lower():
        new_tiling_mode = "snake"
    elif "grid" in tiling_mode.lower():
        new_tiling_mode = "grid"
    else:
        raise ValueError("Unknown tiling mode: " + tiling_mode)
    return new_tiling_mode


def _get_dataset_info_from_config(pipeline_config: dict) -> dict:
    pipeline_config_dict = dict(
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
        + 1,  # id of nuclei channel
        reference_cycle=pipeline_config["channel_names"].index(pipeline_config["nuclei_channel"])
        // (len(pipeline_config["channel_names"]) // pipeline_config["num_cycles"])
        + 1,
        nuclei_channel=pipeline_config["nuclei_channel"],
        membrane_channel=pipeline_config["membrane_channel"],
        nuclei_channel_loc=pipeline_config["nuclei_channel_loc"],
        membrane_channel_loc=pipeline_config["membrane_channel_loc"],
        tiling_mode=_convert_tiling_mode(pipeline_config["tiling_mode"]),
        num_z_planes=1,
        channel_names=pipeline_config["channel_names"],
        channel_names_qc_pass=pipeline_config["channel_names_qc_pass"],
        num_concurrent_tasks=pipeline_config["num_concurrent_tasks"],
        lateral_resolution=pipeline_config["lateral_resolution"],
    )
    return pipeline_config_dict


def load_dataset_info(pipeline_config_path: Path):
    config = load_pipeline_config(pipeline_config_path)
    dataset_info = _get_dataset_info_from_config(config)
    return dataset_info
