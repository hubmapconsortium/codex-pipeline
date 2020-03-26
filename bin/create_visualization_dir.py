#!/usr/bin/env python3

import argparse
from os import walk
from pathlib import Path
import re
from typing import Dict, List
import yaml


def infer_tile_names( cytokitConfig: Dict ) -> List :
    
    tileNames = []
    
    regionHeight, regionWidth = ( 
        cytokitConfig[ "acquisition" ][ "region_height" ],
        cytokitConfig[ "acquisition" ][ "region_width" ]
    )
    regionNames = cytokitConfig[ "acquisition" ][ "region_names" ]
    
    for r in range( 1, len( regionNames ) + 1 ) :
        # Width is X values, height is Y values.
        for x in range( 1, regionWidth + 1 ) :
            for y in range( 1, regionHeight + 1 ) :
                tileNames.append( f"R{r:03}_X{x:03}_Y{y:03}" )

    return tileNames


def collect_target_files(
    tileNames: List,
    cytometryOmeTiffDir: Path,
    expressionsOmeTiffDir: Path,
    sprmResultsDir: Path
) -> Dict :

    targetFiles = {}

    for tile in tileNames :
        
        tileNamePattern = re.compile( tile )

        targetFiles[ tile ] = {}
        
        # Cytokit results files.
        segmOmeTiff = cytometryOmeTiffDir / Path( f"{tile}.ome.tiff" )
        exprsOmeTiff = expressionsOmeTiffDir / Path( f"{tile}.ome.tiff" )
        
        # Make sure the Cytokit results files we need actually exist.
        for f in [ segmOmeTiff, exprsOmeTiff ] :
            if not f.exists() :
                raise FileNotFoundError(
                    f"{f}"
                )
        
        sprmOutputs = []
        for dirpath, dirname, filenames in walk( sprmResultsDir ) :
            for filename in filenames :
                if tileNamePattern.match( filename ) :
                    sprmOutputs.append( sprmResultsDir / Path( filename ) )

        targetFiles[ tile ][ "segm_ome_tiff" ] = segmOmeTiff 
        targetFiles[ tile ][ "exprs_ome_tiff" ] = exprsOmeTiff
        targetFiles[ tile ][ "sprm_outputs" ] = sprmOutputs

    return targetFiles


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
    
    cytokitConfigFile = open( args.cytokit_yaml_config, 'r' )
    cytokitConfig = yaml.safe_load( cytokitConfigFile ) 
    cytokitConfigFile.close()

    tileNames = infer_tile_names( cytokitConfig )
    
    cytometry_ometiff_dir_piece = Path( "cytometry/tile/ome-tiff" )
    expressions_ometiff_dir_piece = Path( "extract/expressions/ome-tiff" )
    
    cytometryOmeTiffDir = args.ometiff_dir / cytometry_ometiff_dir_piece
    expressionsOmeTiffDir = args.ometiff_dir / expressions_ometiff_dir_piece

    targetFiles = collect_target_files(
        tileNames,
        cytometryOmeTiffDir,
        expressionsOmeTiffDir,
        Path( args.sprm_output_dir )
    )

    output_dir = Path( "for-visualization" )
    output_dir.mkdir( parents = True, exist_ok = True )
    
    for tile in targetFiles.keys() :
        
        tileDir = output_dir / Path( tile )
        tileDir.mkdir( parents = True, exist_ok = True )
        
        # TODO: check if this works in CWL pipeline.
        exprsLink = tileDir / Path( "antigen_exprs.ome.tiff" )
        exprsLink.symlink_to( Path( "../../" ) / targetFiles[ tile ][ "exprs_ome_tiff" ] )

        segmLink = tileDir / Path( "segmentation.ome.tiff" )
        segmLink.symlink_to( Path( "../../" ) / targetFiles[ tile ][ "segm_ome_tiff" ] )
        
        tileOmeTiffPattern = re.compile( tile + "\.ome\.tiff-(.*)$" )

        for sprmFile in targetFiles[ tile ][ "sprm_outputs" ] :
            sprmLinkName = tileOmeTiffPattern.match( sprmFile.name ).group( 1 )
            sprmLink = tileDir / Path( sprmLinkName )
            sprmLink.symlink_to( Path( "../../" ) / sprmFile )

