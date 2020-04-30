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

from utils import print_directory_tree

import os

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)


def infer_tile_names( cytokit_config_filename: Path ) -> List[str] :
    
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
    tile_names: List[str],
    directory: Path,
) -> Dict[str, List[Path]] :

    files_by_tile: Dict[str, List[Path]] = defaultdict(list)

    for tile in tile_names :
        tile_name_pattern = re.compile( tile )

        for dirpath_str, dirnames, filenames in walk( directory ) :
            dirpath = Path(dirpath_str)
            for filename in filenames :
                if tile_name_pattern.match( filename ) :
                    files_by_tile[ tile ].append( dirpath / filename )

    # If a tile doesn't have any files, throw an error.
    for tile in tile_names :
        if len( files_by_tile[ tile ] ) == 0 :
            raise ValueError(
                f"No files were found for tile {tile}"
            )

    return files_by_tile

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
        description = (
            "Set up a directory containing the files for the visualization team."
        )
    )
    parser.add_argument(
        "cytokit_yaml_config",
        help = "Path to Cytokit YAML config file.",
        type = Path
    )
    parser.add_argument(
        "expressions_ometiff_dir",
        help = "Path to Cytokit extract output directory from OME-TIFF creation pipeline step.",
        type = Path
    )
    parser.add_argument(
        "cytometry_ometiff_dir",
        help = "Path to Cytokit cytometry output directory from OME-TIFF creation pipeline step.",
        type = Path
    )
    parser.add_argument(
        "sprm_output_dir",
        help = "Path to output directory from SPRM pipeline step.",
        type = Path
    )
    
    args = parser.parse_args()
    
    tile_names = infer_tile_names( args.cytokit_yaml_config )
    
    cytometry_ometiff_dir = args.cytometry_ometiff_dir
    expressions_ometiff_dir = args.expressions_ometiff_dir

    # TODO: use logging for this
    print('Cytometry OME-TIFF directory listing:')
    print_directory_tree(cytometry_ometiff_dir)
    print('Expressions OME-TIFF directory listing:')
    print_directory_tree(expressions_ometiff_dir)
    print('SPRM directory listing:')
    print_directory_tree(args.sprm_output_dir)

    logger.info( args.cytometry_ometiff_dir )
    logger.info( os.listdir( args.cytometry_ometiff_dir ) )
    
    segmentation_mask_ometiffs = collect_files_by_tile( tile_names, cytometry_ometiff_dir )
    expressions_ometiffs = collect_files_by_tile( tile_names, expressions_ometiff_dir )
    sprm_outputs = collect_files_by_tile( tile_names, args.sprm_output_dir )
    
    output_dir = Path( "for-visualization" )
    output_dir.mkdir( parents = True, exist_ok = True )

    output_relative_parent = Path('../..')

    for tile in tile_names :
        tile_dir = output_dir / tile
        tile_dir.mkdir( parents = True, exist_ok = True )

    symlinks_to_archive: List[Tuple[Path, Path]] = []

    # TODO: Perhaps a proper function to do this in a less repetitive way would be nicer.
    for tile in segmentation_mask_ometiffs :
        symlink = output_dir / tile / "segmentation.ome.tiff"
        # There should only be one file here...
        link_target = output_relative_parent / segmentation_mask_ometiffs[ tile ][ 0 ]
        symlinks_to_archive.append((symlink, link_target))
    
    for tile in expressions_ometiffs :
        symlink = output_dir / tile / "antigen_exprs.ome.tiff"
        link_target = output_relative_parent / expressions_ometiffs[ tile ][ 0 ]
        symlinks_to_archive.append((symlink, link_target))

    for tile in sprm_outputs :
        tile_ometiff_pattern = re.compile( tile + "\.ome\.tiff-(.*)$" )
        for sprm_file in sprm_outputs[ tile ] :
            link_name = tile_ometiff_pattern.match( sprm_file.name ).group( 1 )
            link_target = output_relative_parent / sprm_file
            symlink = output_dir / tile / link_name
            symlinks_to_archive.append((symlink, link_target))

    with tarfile.open('symlinks.tar', 'w') as t:
        for symlink, link_target in symlinks_to_archive:
            symlink.symlink_to(link_target)
            logger.info(f'Archiving symlink {symlink} -> {link_target}')
            t.add(symlink)
