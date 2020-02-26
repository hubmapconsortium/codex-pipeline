#!/usr/bin/env python3

from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
import argparse
import logging
import os
import re
import stat
import sys
from tifffile import TiffFile
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_tiff_file_list( directory, tiffFileNamingPattern ) :

    fileList = None
    
    # Get the list of files in the cytometry tile directory.
    try :
        fileList = os.listdir( directory )
    except OSError as err :
        logger.error(
            "Could not acquire list of contents for " +
            args.directory +
            " : " +
            err.strerror
        )
        sys.exit(1)

    # Just take the TIFF files matching the expected naming pattern, in case
    # there are other things in the directory.
    fileList = list(
        filter(
            tiffFileNamingPattern.search,
            fileList
        )
    )

    fileList = [ os.path.join( directory, item ) for item in fileList ]

    return fileList
    

def collect_expressions_extract_channels( extractFile ) :
    
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


def create_ome_tiffs( fileList, tiffFileNamingPattern, outputDir, channelNames ) :
    
    os.mkdir( outputDir )

    for sourceFile in fileList :
        
        logger.info( "Converting file: " + sourceFile )

        fnameMatch = tiffFileNamingPattern.match( os.path.basename( sourceFile ) )

        fname = fnameMatch.group( 1 )
        
        ometiffFilename = os.path.join( outputDir, fname + ".ome.tiff" )
        
        image = AICSImage( sourceFile )
        imageDataForOmeTiff = image.get_image_data( "TCZYX" )

        with ome_tiff_writer.OmeTiffWriter( ometiffFilename ) as ome_writer :
            ome_writer.save(
                imageDataForOmeTiff,
                channel_names = channelNames,
                dimension_order="TCZYX"
            )
        
        logger.info( "OME-TIFF file created: " + ometiffFilename )


########
# MAIN #
########
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
            description = "Convert Cytokit's output TIFFs containing segmentation and extraction results to OME-TIFF, and add the channel names. Creates an \"ome-tiff\" directory inside the output/cytometry/tile and output/extract/expressions directories."
    )
    parser.add_argument(
            "cytokit_output_dir",
            help = "Path to Cytokit's output directory."
    )
    """
    # Commented out until this file is available
    parser.add_argument(
            "antibody_info",
            help = "Path to file containing antibody information"
    )
    """
    
    args = parser.parse_args()
    
    tiffFileNamingPattern = re.compile( r'(^R\d{3}_X\d{3}_Y\d{3})\.tif' )
    
    segmentationChannelNames = [ "cells", "nuclei", "cell_boundaries", "nucleus_boundaries" ]
    
    cytometryTileDir = os.path.join( 
        args.cytokit_output_dir,
        "cytometry",
        "tile"
    )
    
    extractDir = os.path.join(
        args.cytokit_output_dir,
        "extract",
        "expressions"
    )

    segmentationFileList = collect_tiff_file_list( cytometryTileDir, tiffFileNamingPattern )
    
    extractFileList = collect_tiff_file_list( extractDir, tiffFileNamingPattern )

    extractChannelNames = collect_expressions_extract_channels( extractFileList[ 0 ] )

    create_ome_tiffs(
        segmentationFileList,
        tiffFileNamingPattern,
        os.path.join( cytometryTileDir, "ome-tiff" ),
        segmentationChannelNames
    )

    create_ome_tiffs(
        extractFileList,
        tiffFileNamingPattern,
        os.path.join( extractDir, "ome-tiff" ),
        extractChannelNames
    )

