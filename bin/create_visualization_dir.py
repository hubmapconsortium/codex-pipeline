import argparse
import logging
import os
import re
import tarfile
from collections import defaultdict
from os import walk
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from utils import collect_files_by_tile, infer_tile_names, list_directory_tree

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-7s - %(message)s",
)
logger = logging.getLogger(__name__)


def alpha_num_order(string: str) -> str:
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alpha_num_order("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def get_img_listing(in_dir: Path) -> List[Path]:
    allowed_extensions = (".tif", ".tiff")
    listing = list(in_dir.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    img_listing = sorted(img_listing, key=lambda x: alpha_num_order(x.name))
    return img_listing


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def get_file_paths_by_region(dir_listing: List[Path]) -> Dict[int, List[Path]]:
    file_path_by_reg = {}

    for i, path in enumerate(dir_listing):
        region_id = int(re.search(r"^reg(\d+)_*", path.name).groups()[0])
        if region_id in file_path_by_reg:
            file_path_by_reg[region_id] += [path]
        else:
            file_path_by_reg[region_id] = [path]

    return file_path_by_reg


def create_relative_symlink_target(file_path: Path, file_dir: Path, file_symlink: Path) -> Path:

    relative_input_path = file_dir.name / file_path.relative_to(file_dir)
    relative_output_path_piece = Path(*[".."] * (len(file_symlink.parts) - 1))
    return relative_output_path_piece / relative_input_path


def main(cytokit_yaml_config, ometiff_dir, sprm_output_dir):
    cytometry_ometiff_dir = ometiff_dir / "mask"
    expressions_ometiff_dir = ometiff_dir / "expressions"

    # TODO: use logging for this
    logger.debug(f"{ometiff_dir=}")
    logger.debug("Cytometry OME-TIFF directory listing:")
    logger.debug("\n" + list_directory_tree(cytometry_ometiff_dir))
    logger.debug("Expressions OME-TIFF directory listing:")
    logger.debug("\n" + list_directory_tree(expressions_ometiff_dir))
    logger.debug(f"{sprm_output_dir=}")
    logger.debug("SPRM directory listing:")
    logger.debug("\n" + list_directory_tree(sprm_output_dir))

    output_dir = Path("for-visualization")
    make_dir_if_not_exists(output_dir)

    segmentation_mask_ometiffs = get_file_paths_by_region(get_img_listing(cytometry_ometiff_dir))
    expressions_ometiffs = get_file_paths_by_region(get_img_listing(expressions_ometiff_dir))
    sprm_outputs = get_file_paths_by_region(list(sprm_output_dir.iterdir()))

    symlinks_to_archive: List[Tuple[Path, Path]] = []

    # TODO: Perhaps a proper function to do this in a less repetitive way would be nicer.
    for region in segmentation_mask_ometiffs:
        reg_dir = output_dir / Path("reg" + str(region))
        make_dir_if_not_exists(reg_dir)

    for region in segmentation_mask_ometiffs:
        reg_dir = output_dir / Path("reg" + str(region))
        symlink = reg_dir / "segmentation.ome.tiff"
        for img_path in segmentation_mask_ometiffs[region]:
            link_target = create_relative_symlink_target(img_path, ometiff_dir, symlink)
            symlinks_to_archive.append((symlink, link_target))

    for region in expressions_ometiffs:
        reg_dir = output_dir / Path("reg" + str(region))
        symlink = reg_dir / "antigen_exprs.ome.tiff"
        for img_path in expressions_ometiffs[region]:
            link_target = create_relative_symlink_target(img_path, ometiff_dir, symlink)
            symlinks_to_archive.append((symlink, link_target))

    for region in sprm_outputs:
        reg_dir = output_dir / Path("reg" + str(region))
        for sprm_file_path in sprm_outputs[region]:
            symlink = reg_dir / sprm_file_path.name
            link_target = create_relative_symlink_target(sprm_file_path, sprm_output_dir, symlink)
            symlinks_to_archive.append((symlink, link_target))

    with tarfile.open("symlinks.tar", "w") as t:
        for symlink, link_target in symlinks_to_archive:
            symlink.symlink_to(link_target)
            logger.info(f"Archiving symlink {symlink} -> {link_target}")
            t.add(symlink)

    for symlink, link_target in symlinks_to_archive:
        symlink.unlink()


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
    main(args.cytokit_yaml_config, args.ometiff_dir, args.sprm_output_dir)
