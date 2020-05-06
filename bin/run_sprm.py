#!/usr/bin/env python3

import argparse
import logging
from os import fspath
from pathlib import Path

from utils import list_directory_tree

logging.basicConfig(
    level=logging.DEBUG,
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
        "ometiff_dir",
        help = "Path to directory containing OME-TIFF files.",
        type = Path
    )
    args = parser.parse_args()

    expressions_ometiff_dir = args.ometiff_dir / 'extract/expressions/ome-tiff'
    cytometry_ometiff_dir = args.ometiff_dir / 'cytometry/tile/ome-tiff'

    logger.debug('Expressions OME-TIFF directory:')
    logger.debug(list_directory_tree(expressions_ometiff_dir))
    logger.debug('Cytometry OME-TIFF directory:')
    logger.debug(list_directory_tree(cytometry_ometiff_dir))

    import sys
    sys.path.append( fspath(SPRM_DIR) )
    import SPRM

    # Run SPRM.
    logger.info( "Running SPRM ..." )

    sprm_output_dir = Path("sprm_outputs")
    try :
        SPRM.main(
            expressions_ometiff_dir,
            cytometry_ometiff_dir,
            sprm_output_dir,
            SPRM_DIR / "options.txt",
        )
    except Exception as e :
        logger.exception( "SPRM failed." )
    else :
        logger.info( "SPRM completed." )

    logger.debug('Output:')
    logger.debug(list_directory_tree(sprm_output_dir))
