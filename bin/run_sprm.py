#!/usr/bin/env python3

import argparse
import logging
from os import fspath
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)

SPRM_DIR = Path('/opt/sprm')

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
        description = (
            "Wrapper script for SPRM analysis code."
        )
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

    import sys
    sys.path.append( fspath('/opt/sprm') )
    import SPRM

    # Run SPRM.
    logger.info( "Running SPRM ..." )

    try :
        SPRM.main(
            fspath( args.expressions_ometiff_dir ),
            fspath( args.cytometry_ometiff_dir ),
            fspath( SPRM_DIR / "options.txt" )
        )
    except Exception as e :
        logger.exception( "SPRM failed." )
    else :
        logger.info( "SPRM completed." )

    sprm_dir = Path( "results" )
    sprm_dir.rename( "sprm_outputs" )
