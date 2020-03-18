#!/use/bin/env python3

import argparse
from pathlib import Path
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
    cellshapesDir: Path
) -> Dict :

    targetFiles = {}

    for tile in tileNames :
        
        targetFiles[ tile ] = {}

        segmOmeTiff = cytometryOmeTiffDir / Path( f"{tile}.ome.tiff" )
        exprsOmeTiff = expressionsOmeTiffDir / Path( f"{tile}.ome.tiff" )
        cellCsv = cellshapesDir / Path( f"{tile}.shape.csv" )
        
        # Make sure all the files we need actually exist.
        for f in [ segmOmeTiff, exprsOmeTiff, cellCsv ] :
            if not f.exists() :
                raise FileNotFoundError(
                    f"{f}"
                )
        
        targetFiles[ tile ][ "segm_ome_tiff" ] = segmOmeTiff 
        targetFiles[ tile ][ "exprs_ome_tiff" ] = exprsOmeTiff
        targetFiles[ tile ][ "cellshapes_csv" ] = cellCsv

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
        "cellshapes_output_dir",
        help = "Path to output directory from cell shape CSV creation pipeline step.",
        type = Path
    )
    
    args = parser.parse_args()
    
    cytokitConfigFile = open( args.cytokit_yaml_config, 'r' )
    cytokitConfig = yaml.safe_load( cytokitConfigFile ) 
    cytokitConfigFile.close()

    tileNames = infer_tile_names( cytokitConfig )
    
    cytometry_ometiff_dir_piece = Path( "cytometry/tile/ome-tiff" )
    expressions_ometiff_dir_piece = Path( "extract/expressions/ome-tiff" )
    cellshapes_dir_piece = Path( "cytometry/statistics/cellshapes" )

    cytometryOmeTiffDir = args.ometiff_dir / cytometry_ometiff_dir_piece
    expressionsOmeTiffDir = args.ometiff_dir / expressions_ometiff_dir_piece
    cellshapesDir = args.cellshapes_output_dir / cellshapes_dir_piece
    
    targetFiles = collect_target_files(
        tileNames,
        cytometryOmeTiffDir,
        expressionsOmeTiffDir,
        cellshapesDir
    )

    output_dir = Path( "for-visualization" )
    output_dir.mkdir( parents = True, exist_ok = True )
    
    for tile in targetFiles.keys() :

        tileDir = output_dir / Path( tile )
        tileDir.mkdir( parents = True, exist_ok = True )

        exprsLink = tileDir / Path( "antigen_exprs.ome.tiff" )
        exprsLink.symlink_to( targetFiles[ tile ][ "exprs_ome_tiff" ] )

        segmLink = tileDir / Path( "segmentation.ome.tiff" )
        segmLink.symlink_to( targetFiles[ tile ][ "segm_ome_tiff" ] )

        cellCsvLink = tileDir / Path( "cell_spatial.csv" )
        cellCsvLink.symlink_to( targetFiles[ tile][ "cellshapes_csv" ] )


