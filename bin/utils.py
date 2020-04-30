from os import walk
from pathlib import Path
from pprint import pformat, pprint
import re
from typing import Dict, List
import yaml


def list_directory_tree(directory: Path) -> str:
    return pformat(sorted(directory.glob('**/*')))


def print_directory_tree(directory: Path):
    print(list_directory_tree(directory))


def infer_tile_names( cytokit_config_filename: Path ) -> List :

    cytokit_config_file = open( cytokit_config_filename, 'r' )
    cytokit_config = yaml.safe_load( cytokit_config_file )
    cytokit_config_file.close()

    tile_names = []

    region_height, region_width = (
        cytokit_config[ "acquisition" ][ "region_height" ],
        cytokit_config[ "acquisition" ][ "region_width" ]
    )
    region_names = cytokit_config[ "acquisition" ][ "region_names" ]

    for r in range( 1, len( region_names ) + 1 ) :
        # Width is X values, height is Y values.
        for x in range( 1, region_width + 1 ) :
            for y in range( 1, region_height + 1 ) :
                tile_names.append( f"R{r:03}_X{x:03}_Y{y:03}" )

    return tile_names


def collect_files_by_tile(
    tile_names: List,
    directory: Path
) -> Dict :

    files_by_tile = {}

    for tile in tile_names :

        files_by_tile[ tile ] = []

        tile_name_pattern = re.compile( tile )

        for dirpath, dirnames, filenames in walk( directory ) :
            for filename in filenames :
                if tile_name_pattern.match( filename ) :
                    files_by_tile[ tile ].append( directory / Path( filename ) )

    # If a tile doesn't have any files, throw an error.
    for tile in tile_names :
        if len( files_by_tile[ tile ] ) == 0 :
            raise ValueError(
                f"No files were found for tile {tile}"
            )
            return

    return files_by_tile
