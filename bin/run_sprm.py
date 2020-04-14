#!/usr/bin/env python3

import argparse
import logging
from os import chdir, environ, remove
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
        description = (
            "Wrapper script for SPRM analysis code."
        )
    )
    parser.add_argument(
        "sprm_dir",
        help = "Path to SPRM checkout.",
        type = Path
    )
    parser.add_argument(
        "expressions_ometiff_dir",
        help = "Path to directory containing OME-TIFF files with expression intensities.",
        type = Path
    )
    parser.add_argument(
        "cytometry_ometiff_dir",
        help = "Path to directory containing OME-TIFF files with Cytokit segmentation masks in.",
        type = Path
    )

    args = parser.parse_args()

    # Set up an output directory and move to it.
    output_dir = Path( "sprm_outputs" )
    output_dir.mkdir( parents = True, exist_ok = True )
    chdir( output_dir )
    
    import sys
    sys.path.append( str( args.sprm_dir ) )
    import SPRM

    # Run SPRM.
    logger.info( "Running SPRM ..." )
    
    try :
        SPRM.main(
            args.expressions_ometiff_dir,
            args.cytometry_ometiff_dir,
            args.sprm_dir / Path( "options.txt" )
        )
    except Exception as e :
        logger.error( f"SPRM failed: {e}" )
    else :
        logger.info( "SPRM completed." )
