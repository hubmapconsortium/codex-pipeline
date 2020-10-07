from pathlib import Path
import json
from typing import Tuple


def generate_slicer_info(block_shape_no_overlap: Tuple[int, int], overlap: int, stitched_img_shape: Tuple[int, int]) -> dict:
    slicer_info = dict()
    slicer_info['slicer'] = dict()

    img_height, img_width = stitched_img_shape
    block_height, block_width = block_shape_no_overlap

    padding = dict(left=0, right=0, top=0, bottom=0)
    padding["right"] = block_width - (img_width % block_width)
    padding["bottom"] = block_height - (img_height % block_height)

    x_nblocks = img_width // block_width if img_width % block_width == 0 else (img_width // block_width) + 1
    y_nblocks = img_height // block_height if img_height % block_height == 0 else (img_height // block_height) + 1

    slicer_info['slicer']['padding'] = padding
    slicer_info['slicer']['overlap'] = overlap
    slicer_info['slicer']['num_blocks'] = {'x': x_nblocks, 'y': y_nblocks}
    slicer_info['slicer']['block_shape_no_overlap'] = {'x': block_width, 'y': block_height}
    slicer_info['slicer']['block_shape_with_overlap'] = {'x': block_width + overlap * 2, 'y': block_height + overlap * 2}
    return slicer_info


def replace_values_in_config(exp, slicer_info):

    values_to_replace = {'tiling_mode': 'grid',
                         'region_width': slicer_info['slicer']['num_blocks']['x'],
                         'region_height': slicer_info['slicer']['num_blocks']['y'],
                         'num_z_planes': 1,
                         'tile_width': slicer_info['slicer']['block_shape_no_overlap']['x'],
                         'tile_height': slicer_info['slicer']['block_shape_no_overlap']['y'],
                         'tile_overlap_x': slicer_info['slicer']['overlap'] * 2,
                         'tile_overlap_y': slicer_info['slicer']['overlap'] * 2,
                         'target_shape': [slicer_info['slicer']['block_shape_no_overlap']['x'],
                                          slicer_info['slicer']['block_shape_no_overlap']['y']]
                         }

    exp.update(values_to_replace)
    return exp


def modify_pipeline_config(path_to_config: Path, block_shape_no_overlap: Tuple[int, int], overlap: int, stitched_img_shape: Tuple[int, int]):
    with open(path_to_config, 'r') as s:
        config = json.load(s)

    slicer_info = generate_slicer_info(block_shape_no_overlap, overlap, stitched_img_shape)
    config = replace_values_in_config(config, slicer_info)
    config.update(slicer_info)

    return config
