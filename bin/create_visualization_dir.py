#!/usr/bin/env python3

import argparse
from collections import defaultdict
import logging
from os import walk
from pathlib import Path
import re
import tarfile
from typing import Dict, List, Tuple
import yaml

from utils import collect_files_by_tile, infer_tile_names, list_directory_tree

import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-7s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_relative_symlink_target(
    input_file_path: Path,
    input_base_path: Path,
    relative_output_path: Path,
) -> Path:
    relative_input_path = input_base_path.name / input_file_path.relative_to(input_base_path)
    relative_output_path_piece = Path(*[".."] * (len(relative_output_path.parts) - 1))
    return relative_output_path_piece / relative_input_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Set up a directory containing the files for the visualization team.")
    )
    parser.add_argument(
        "cytokit_yaml_config",
        help="Path to Cytokit YAML config file.",
        type=Path,
    )
    parser.add_argument(
        "ometiff_dir",
        help="Path to Cytokit output directory from OME-TIFF creation pipeline step.",
        type=Path,
    )
    parser.add_argument(
        "sprm_output_dir",
        help="Path to output directory from SPRM pipeline step.",
        type=Path,
    )

    args = parser.parse_args()

    tile_names = infer_tile_names(args.cytokit_yaml_config)

    cytometry_ometiff_dir = args.ometiff_dir / "cytometry/tile/ome-tiff"
    expressions_ometiff_dir = args.ometiff_dir / "extract/expressions/ome-tiff"

    # TODO: use logging for this
    logger.debug(f"{args.ometiff_dir=}")
    logger.debug("Cytometry OME-TIFF directory listing:")
    logger.debug("\n" + list_directory_tree(cytometry_ometiff_dir))
    logger.debug("Expressions OME-TIFF directory listing:")
    logger.debug("\n" + list_directory_tree(expressions_ometiff_dir))
    logger.debug(f"{args.sprm_output_dir=}")
    logger.debug("SPRM directory listing:")
    logger.debug("\n" + list_directory_tree(args.sprm_output_dir))

    segmentation_mask_ometiffs = collect_files_by_tile(tile_names, cytometry_ometiff_dir)
    expressions_ometiffs = collect_files_by_tile(tile_names, expressions_ometiff_dir)
    sprm_outputs = collect_files_by_tile(tile_names, args.sprm_output_dir, allow_empty_tiles=True)

    output_dir = Path("for-visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile in tile_names:
        tile_dir = output_dir / tile
        tile_dir.mkdir(parents=True, exist_ok=True)

    symlinks_to_archive: List[Tuple[Path, Path]] = []

    # TODO: Perhaps a proper function to do this in a less repetitive way would be nicer.
    for tile in segmentation_mask_ometiffs:
        symlink = output_dir / tile / "segmentation.ome.tiff"
        # There should only be one file here...
        link_target = create_relative_symlink_target(
            segmentation_mask_ometiffs[tile][0],
            args.ometiff_dir,
            symlink,
        )
        symlinks_to_archive.append((symlink, link_target))

    for tile in expressions_ometiffs:
        symlink = output_dir / tile / "antigen_exprs.ome.tiff"
        link_target = create_relative_symlink_target(
            expressions_ometiffs[tile][0],
            args.ometiff_dir,
            symlink,
        )
        symlinks_to_archive.append((symlink, link_target))

    for tile in sprm_outputs:
        tile_ometiff_pattern = re.compile(tile + "\.ome\.tiff-(.*)$")
        for sprm_file in sprm_outputs[tile]:
            link_name = tile_ometiff_pattern.match(sprm_file.name).group(1)
            symlink = output_dir / tile / link_name
            link_target = create_relative_symlink_target(
                sprm_file,
                args.sprm_output_dir,
                symlink,
            )
            symlinks_to_archive.append((symlink, link_target))

    with tarfile.open("symlinks.tar", "w") as t:
        for symlink, link_target in symlinks_to_archive:
            symlink.symlink_to(link_target)
            logger.info(f"Archiving symlink {symlink} -> {link_target}")
            t.add(symlink)

    for symlink, link_target in symlinks_to_archive:
        symlink.unlink()
