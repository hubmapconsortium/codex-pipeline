import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _get_dataset_info_from_config(pipeline_config: dict) -> Dict[str, Any]:
    required_fields: List[Tuple[str, Optional[str]]] = [
        ("num_cycles", None),
        ("num_tiles_x", "region_width"),
        ("num_tiles_y", "region_height"),
        ("tile_width", None),
        ("tile_height", None),
        ("tile_dtype", None),
        ("overlap_x", "tile_overlap_x"),
        ("overlap_y", "tile_overlap_y"),
        ("pixel_distance_x", "lateral_resolution"),
        ("pixel_distance_y", "lateral_resolution"),
        ("pixel_distance_z", "axial_resolution"),
        ("nuclei_channel", None),
        ("membrane_channel", None),
        ("nuclei_channel_loc", None),
        ("membrane_channel_loc", None),
        ("num_z_planes", None),
        ("channel_names", None),
        ("channel_names_qc_pass", None),
        ("num_concurrent_tasks", None),
        ("lateral_resolution", None),
    ]
    optional_fields: List[Tuple[str, Optional[str]]] = [
        ("membrane_channel", None),
    ]
    pipeline_config_dict = dict(
        dataset_dir=Path(pipeline_config["raw_data_location"]),
        num_channels=len(pipeline_config["channel_names"]) // pipeline_config["num_cycles"],
        num_tiles=pipeline_config["region_width"] * pipeline_config["region_height"],
        # does not matter because we have only one z-plane:
        overlap_z=1,
        # id of nuclei channel:
        reference_channel=pipeline_config["channel_names"].index(pipeline_config["nuclei_channel"])
        + 1,
        reference_cycle=pipeline_config["channel_names"].index(pipeline_config["nuclei_channel"])
        // (len(pipeline_config["channel_names"]) // pipeline_config["num_cycles"])
        + 1,
        tiling_mode=_convert_tiling_mode(pipeline_config["tiling_mode"]),
    )
    for field, source in required_fields:
        if source is None:
            source = field
        pipeline_config_dict[field] = pipeline_config[source]
    for field, source in optional_fields:
        if source is None:
            source = field
        if source in pipeline_config:
            pipeline_config_dict[field] = pipeline_config[source]
    return pipeline_config_dict


def load_dataset_info(pipeline_config_path: Path):
    config = load_pipeline_config(pipeline_config_path)
    dataset_info = _get_dataset_info_from_config(config)
    return dataset_info
