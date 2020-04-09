#!/usr/bin/env python3

import argparse
import logging
from os import chdir, remove
from pathlib import Path
from subprocess import check_call, CalledProcessError

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
        "sprm_script",
        help = "Path to SPRM.py",
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

    # Run SPRM.
    logger.info( "Running SPRM ..." )
    sprm_command = [
        "python",
        str( args.sprm_script ),
        str( args.expressions_ometiff_dir ),
        str( args.cytometry_ometiff_dir )
    ]
    try :
        check_call( sprm_command )
    except CalledProcessError as e :
        logger.error( f"SPRM failed: {e}" )
    else :
        logger.info( "SPRM completed." )
