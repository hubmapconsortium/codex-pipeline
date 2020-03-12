#!/usr/bin/env python3

from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
import argparse
import logging
from multiprocessing import Pool
from os import walk
from pathlib import Path
import re
from tifffile import TiffFile
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)

SEGMENTATION_CHANNEL_NAMES = [
    "cells",
    "nuclei",
    "cell_boundaries",
    "nucleus_boundaries",
]

TIFF_FILE_NAMING_PATTERN = re.compile( r'^R\d{3}_X\d{3}_Y\d{3}\.tif' )


def collect_tiff_file_list(
        directory: Path,
        TIFF_FILE_NAMING_PATTERN: re.Pattern
) -> List[ Path ] :

    fileList = []

    for dirpath, dirnames, filenames in walk( directory ) :
        for filename in filenames :
            if TIFF_FILE_NAMING_PATTERN.match( filename ) :
                fileList.append( directory / filename )

    if len( fileList ) == 0 :
        logger.warning( "No files found in " + str( directory ) )

    return fileList


def collect_expressions_extract_channels(extractFile: Path) -> List[str]:
    """
    Read file with TiffFile to get Labels attribute from ImageJ metadata. We
    need this to get the channel names in the correct order. Cytokit re-orders
    them compared to the order in the YAML config.  The ImageJ "Labels"
    attribute isn't picked up by AICSImageIO.
    """
    img = TiffFile( extractFile )

    numChannels = int( img.imagej_metadata[ "channels" ] )

    channelList = img.imagej_metadata[ "Labels" ][ 0:numChannels ]

    # Remove "proc_" from the start of the channel names.
    procPattern = re.compile( r'^proc_(.*)' )

    channelList = [ procPattern.match( channel ).group( 1 ) for channel in channelList ]

    return channelList


def convert_tiff_file(
        filesAndChannels: Tuple[ Path, Path, List ]
) :

    sourceFile, ometiffFile, channelNames = filesAndChannels

    logger.info( "Converting file: " + str( sourceFile ) )

    image = AICSImage( sourceFile )

    imageDataForOmeTiff = image.get_image_data( "TCZYX" )

    with ome_tiff_writer.OmeTiffWriter( ometiffFile ) as ome_writer :
        ome_writer.save(
            imageDataForOmeTiff,
            channel_names = channelNames,
            dimension_order="TCZYX"
        )

    logger.info( "OME-TIFF file created: " + str( ometiffFile ) )


def create_ome_tiffs(
        file_list: List[Path],
        output_dir: Path,
        channel_names: List[str],
        subprocesses: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files_and_channels = []

    for source_file in file_list:
        ome_tiff_file = (output_dir / source_file.name).with_suffix(".ome.tiff")
        all_files_and_channels.append((source_file, ome_tiff_file, channel_names))

    with Pool(processes=subprocesses) as pool:
        pool.imap_unordered(convert_tiff_file, all_files_and_channels)
        pool.close()
        pool.join()



########
# MAIN #
########
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
        description=(
            "Convert Cytokit's output TIFFs containing segmentation and extraction "
            "results to OME-TIFF, and add the channel names. Creates an \"ome-tiff\" "
            "directory inside the output/cytometry/tile and "
            "output/extract/expressions directories."
        ),
    )
    parser.add_argument(
        "cytokit_output_dir",
        help="Path to Cytokit's output directory.",
        type=Path,
    )
    parser.add_argument(
        '-p',
        '--processes',
        help='Number of parallel OME-TIFF conversions to perform at once',
        type=int,
        default=8,
    )
    """
    # Commented out until this file is available
    parser.add_argument(
            "antibody_info",
            help = "Path to file containing antibody information"
    )
    """

    args = parser.parse_args()

    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)

    cytometry_tile_dir_piece = Path("cytometry/tile")
    extract_expressions_piece = Path("extract/expressions")

    cytometryTileDir = args.cytokit_output_dir / cytometry_tile_dir_piece
    extractDir = args.cytokit_output_dir / extract_expressions_piece

    segmentationFileList = collect_tiff_file_list( cytometryTileDir, TIFF_FILE_NAMING_PATTERN )
    extractFileList = collect_tiff_file_list( extractDir, TIFF_FILE_NAMING_PATTERN )

    if segmentationFileList:
        create_ome_tiffs(
            segmentationFileList,
            output_dir / cytometry_tile_dir_piece / 'ome-tiff',
            SEGMENTATION_CHANNEL_NAMES,
            args.processes,
        )

    if extractFileList:
        # For the extract, pull the correctly ordered list of channel names from
        # one of the files, as they aren't guaranteed to be in the same order as
        # the YAML config.
        extractChannelNames = collect_expressions_extract_channels( extractFileList[ 0 ] )

        create_ome_tiffs(
            extractFileList,
            output_dir / extract_expressions_piece / 'ome-tiff',
            extractChannelNames,
            args.processes,
        )

