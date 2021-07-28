import json
from pathlib import Path
from typing import Tuple


def generate_slicer_info(
    tile_shape_no_overlap: Tuple[int, int], overlap: int, stitched_img_shape: Tuple[int, int]
) -> dict:
    slicer_info = dict()
    slicer_info["slicer"] = dict()

    img_height, img_width = stitched_img_shape
    tile_height, tile_width = tile_shape_no_overlap

    padding = dict(left=0, right=0, top=0, bottom=0)
    if img_width % tile_width == 0:
        padding["right"] = 0
    else:
        padding["right"] = tile_width - (img_width % tile_width)
    if img_height % tile_height == 0:
        padding["bottom"] = 0
    else:
        padding["bottom"] = tile_height - (img_height % tile_height)

    x_ntiles = (
        img_width // tile_width if img_width % tile_width == 0 else (img_width // tile_width) + 1
    )
    y_ntiles = (
        img_height // tile_height
        if img_height % tile_height == 0
        else (img_height // tile_height) + 1
    )

    slicer_info["slicer"]["padding"] = padding
    slicer_info["slicer"]["overlap"] = overlap
    slicer_info["slicer"]["num_tiles"] = {"x": x_ntiles, "y": y_ntiles}
    slicer_info["slicer"]["tile_shape_no_overlap"] = {"x": tile_width, "y": tile_height}
    slicer_info["slicer"]["tile_shape_with_overlap"] = {
        "x": tile_width + overlap * 2,
        "y": tile_height + overlap * 2,
    }
    return slicer_info


def replace_values_in_config(exp, slicer_info):
    original_measurements = {
        "original_measurements": {
            "tiling_mode": exp["tiling_mode"],
            "region_width": exp["region_width"],
            "region_height": exp["region_height"],
            "num_z_planes": exp["num_z_planes"],
            "tile_width": exp["tile_width"],
            "tile_height": exp["tile_height"],
            "tile_overlap_x": exp["tile_overlap_x"],
            "tile_overlap_y": exp["tile_overlap_y"],
            "target_shape": exp["target_shape"],
        }
    }
    values_to_replace = {
        "tiling_mode": "grid",
        "region_width": slicer_info["slicer"]["num_tiles"]["x"],
        "region_height": slicer_info["slicer"]["num_tiles"]["y"],
        "num_z_planes": 1,
        "tile_width": slicer_info["slicer"]["tile_shape_no_overlap"]["x"],
        "tile_height": slicer_info["slicer"]["tile_shape_no_overlap"]["y"],
        "tile_overlap_x": slicer_info["slicer"]["overlap"] * 2,
        "tile_overlap_y": slicer_info["slicer"]["overlap"] * 2,
        "target_shape": [
            slicer_info["slicer"]["tile_shape_no_overlap"]["x"],
            slicer_info["slicer"]["tile_shape_no_overlap"]["y"],
        ],
    }

    exp.update(values_to_replace)
    exp.update(original_measurements)
    return exp


def modify_pipeline_config(
    path_to_config: Path,
    tile_shape_no_overlap: Tuple[int, int],
    overlap: int,
    stitched_img_shape: Tuple[int, int],
):
    with open(path_to_config, "r") as s:
        config = json.load(s)

    slicer_info = generate_slicer_info(tile_shape_no_overlap, overlap, stitched_img_shape)
    config = replace_values_in_config(config, slicer_info)
    config.update(slicer_info)

    return config


def save_modified_pipeline_config(pipeline_config: dict, out_dir: Path):
    out_file_path = out_dir.joinpath("pipelineConfig.json")
    with open(out_file_path, "w") as s:
        json.dump(pipeline_config, s, indent=4)
