from collections import defaultdict
from os import walk
from pathlib import Path
from pprint import pformat
import re
from typing import Dict, List

import yaml


def list_directory_tree(directory: Path) -> str:
    return pformat(sorted(directory.glob("**/*"))) + "\n"


def print_directory_tree(directory: Path):
    print(list_directory_tree(directory))


def infer_tile_names(cytokit_config_filename: Path) -> List[str]:
    with open(cytokit_config_filename) as cytokit_config_file:
        cytokit_config = yaml.safe_load(cytokit_config_file)

    tile_names = []

    region_height, region_width = (
        cytokit_config["acquisition"]["region_height"],
        cytokit_config["acquisition"]["region_width"],
    )
    region_names = cytokit_config["acquisition"]["region_names"]

    for r in range(1, len(region_names) + 1):
        # Width is X values, height is Y values.
        for x in range(1, region_width + 1):
            for y in range(1, region_height + 1):
                tile_names.append(f"R{r:03}_X{x:03}_Y{y:03}")

    return tile_names


def collect_files_by_tile(
    tile_names: List[str],
    directory: Path,
    *,
    allow_empty_tiles: bool = False,
) -> Dict[str, List[Path]]:

    files_by_tile: Dict[str, List[Path]] = defaultdict(list)

    for tile in tile_names:
        tile_name_pattern = re.compile(tile)

        for dirpath_str, dirnames, filenames in walk(directory):
            dirpath = Path(dirpath_str)
            for filename in filenames:
                if tile_name_pattern.match(filename):
                    files_by_tile[tile].append(dirpath / filename)

    # If a tile doesn't have any files, throw an error unless explicitly allowed.
    if not allow_empty_tiles:
        for tile in tile_names:
            if len(files_by_tile[tile]) == 0:
                raise ValueError(f"No files were found for tile {tile}")

    return files_by_tile
