import argparse
import logging
from pathlib import Path

from sprm import SPRM
from utils import list_directory_tree

logging.basicConfig(level=logging.DEBUG, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Wrapper script for SPRM analysis code.",
    )
    parser.add_argument(
        "ometiff_dir",
        help="Path to directory containing OME-TIFF files.",
        type=Path,
    )
    args = parser.parse_args()
    
    expressions_ometiff_dir = args.ometiff_dir / 'expressions'
    cytometry_ometiff_dir = args.ometiff_dir / 'mask'


    logger.debug("Expressions OME-TIFF directory:")
    logger.debug(list_directory_tree(expressions_ometiff_dir))
    logger.debug("Cytometry OME-TIFF directory:")
    logger.debug(list_directory_tree(cytometry_ometiff_dir))

    # Run SPRM.
    logger.info("Running SPRM ...")

    sprm_output_dir = Path("sprm_outputs")
    SPRM.main(
        expressions_ometiff_dir,
        cytometry_ometiff_dir,
        output_dir=sprm_output_dir,
    )
    logger.info("SPRM completed.")

    logger.debug("Output:")
    logger.debug(list_directory_tree(sprm_output_dir))
