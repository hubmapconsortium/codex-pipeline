#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
from typing import Dict, List

from utils import infer_tile_names, collect_files_by_tile


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
        "ometiff_dir",
        help = "Path to Cytokit output directory from OME-TIFF creation pipeline step.",
        type = Path
    )
    parser.add_argument(
        "sprm_output_dir",
        help = "Path to output directory from SPRM pipeline step.",
        type = Path
    )
    
    args = parser.parse_args()
    
    tile_names = infer_tile_names( args.cytokit_yaml_config )
    
    cytometry_ometiff_dir_piece = Path( "cytometry/tile/ome-tiff" )
    expressions_ometiff_dir_piece = Path( "extract/expressions/ome-tiff" )
    
    cytometry_ometiff_dir = args.ometiff_dir / cytometry_ometiff_dir_piece
    expressions_ometiff_dir = args.ometiff_dir / expressions_ometiff_dir_piece
    
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

