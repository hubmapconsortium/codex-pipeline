from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
import argparse
import logging
import os
import re
import stat
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
            description = "Convert Cytokit's output TIFFs to OME-TIFF. For segmentation results, add the channel names only. For processed images, remove \"blank\" and \"empty\" channels, duplicated nuclei channels (retaining the one used for nuclei segmentation), and add channel names based on Cytokit YAML config."
    )
    parser.add_argument(
            "cytometry_tile_dir",
            help = "Path to Cytokits output/cytometry/tile directory."
    )
    parser.add_argument(
            "output_dir",
            help = "Path to directory to write OME-TIFF files to."
    )


    channelNames = [ "cells", "nuclei", "cell_boundaries", "nucleus_boundaries" ]
    tiffFileNamingPattern = re.compile( r'(^R\d{3}_X\d{3}_Y\d{3})\.tif' )


    args = parser.parse_args()

    fileList = None
    
    # Get the list of files in the cytometry tile directory.
    try :
        fileList = os.listdir( args.cytometry_tile_dir )
    except OSError as err :
        logger.error(
            "Could not acquire list of contents for " +
            args.cytometry_tile_dir +
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


    for f in fileList :
        
        sourceFile = os.path.join( args.cytometry_tile_dir, f )
        
        logger.info( "Converting file: " + sourceFile )

        fnameMatch = tiffFileNamingPattern.match( f )

        fname = fnameMatch.group( 1 )
        
        ometiffFilename = os.path.join( args.output_dir, fname + ".ome.tiff" )
        
        image = AICSImage( sourceFile )
        imageDataForOmeTiff = image.get_image_data( "TCZYX" )

        with ome_tiff_writer.OmeTiffWriter( ometiffFilename ) as ome_writer :
            ome_writer.save(
                imageDataForOmeTiff,
                channel_names = channelNames,
                dimension_order="TCZYX"
            )
        
        
        logger.info( "OME-TIFF file created: " + ometiffFilename )

