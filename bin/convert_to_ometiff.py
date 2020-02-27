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
        "nucleus_boundaries" 
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
    

def collect_expressions_extract_channels( 
        extractFile: Path 
) -> List[ str ] :
    
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
    
    ome_writer = ome_tiff_writer.OmeTiffWriter( ometiffFile )
    
    with ome_tiff_writer.OmeTiffWriter( ometiffFile ) as ome_writer :
        ome_writer.save(
            imageDataForOmeTiff,
            channel_names = channelNames,
            dimension_order="TCZYX"
        )

    logger.info( "OME-TIFF file created: " + str( ometiffFile ) )


def create_ome_tiffs( 
        fileList: List[ Path ], 
        outputDir: Path, 
        channelNames: List[ str ] 
) :
    
    Path.mkdir( outputDir )
    
    allFilesAndChannels = []

    for sourceFile in fileList :
        ometiffFile = ( outputDir / sourceFile.name ).with_suffix( ".ome.tiff" )
        allFilesAndChannels.append( ( sourceFile, ometiffFile, channelNames ) )
    
    with Pool( processes = 8 ) as pool :
        pool.imap_unordered( convert_tiff_file, allFilesAndChannels )
        pool.close()
        pool.join()



########
# MAIN #
########
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
            description = "Convert Cytokit's output TIFFs containing segmentation and extraction results to OME-TIFF, and add the channel names. Creates an \"ome-tiff\" directory inside the output/cytometry/tile and output/extract/expressions directories."
    )
    parser.add_argument(
            "cytokit_output_dir",
            help = "Path to Cytokit's output directory.",
            type = Path
    )
    """
    # Commented out until this file is available
    parser.add_argument(
            "antibody_info",
            help = "Path to file containing antibody information"
    )
    """
    
    args = parser.parse_args()
    
    cytometryTileDir = args.cytokit_output_dir / "cytometry" / "tile"
    extractDir = args.cytokit_output_dir / "extract" / "expressions"
    
    segmentationFileList = collect_tiff_file_list( cytometryTileDir, TIFF_FILE_NAMING_PATTERN )
    extractFileList = collect_tiff_file_list( extractDir, TIFF_FILE_NAMING_PATTERN )

    # For the extract, pull the correctly ordered list of channel names from
    # one of the files, as they aren't guaranteed to be in the same order as
    # the YAML config.
    extractChannelNames = collect_expressions_extract_channels( extractFileList[ 0 ] )
    
    if len( segmentationFileList ) > 0 :
        create_ome_tiffs(
            segmentationFileList,
            cytometryTileDir / "ome-tiff",
            SEGMENTATION_CHANNEL_NAMES
        )
    
    if len( extractFileList ) > 0 :
        create_ome_tiffs(
            extractFileList,
            extractDir / "ome-tiff",
            extractChannelNames
        )

