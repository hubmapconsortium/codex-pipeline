#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, List
import yaml

#from utils import infer_tile_names, collect_files_by_tile

import os

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    logger.info( args.cytometry_ometiff_dir )
    logger.info( os.listdir( args.cytometry_ometiff_dir ) )
    
    segmentation_mask_ometiffs = collect_files_by_tile( tile_names, cytometry_ometiff_dir )
    expressions_ometiffs = collect_files_by_tile( tile_names, expressions_ometiff_dir )
    sprm_outputs = collect_files_by_tile( tile_names, args.sprm_output_dir )
    
    output_dir = Path( "for-visualization" )
    output_dir.mkdir( parents = True, exist_ok = True )
    
    for tile in tile_names :
        tile_dir = output_dir / Path( tile )
        tile_dir.mkdir( parents = True, exist_ok = True )
        
    # TODO: Perhaps a proper function to do this in a less repetitive way would be nicer.
    for tile in segmentation_mask_ometiffs :
        symlink = output_dir / Path( tile ) / Path( "segmentation.ome.tiff" )
        # There should only be one file here...
        symlink.symlink_to( Path( "../.." ) / segmentation_mask_ometiffs[ tile ][ 0 ] )
    
    for tile in expressions_ometiffs :
        symlink = output_dir / Path( tile ) / Path( "antigen_exprs.ome.tiff" )
        symlink.symlink_to( Path( "../.." ) / expressions_ometiffs[ tile ][ 0 ] )

    for tile in sprm_outputs :
        tile_ometiff_pattern = re.compile( tile + "\.ome\.tiff-(.*)$" )
        for sprm_file in sprm_outputs[ tile ] :
            link_name = tile_ometiff_pattern.match( sprm_file.name ).group( 1 )
            symlink = output_dir / Path( tile ) / Path( link_name )
            symlink.symlink_to( Path( "../.." ) / sprm_file )

