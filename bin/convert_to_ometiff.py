#!/usr/bin/env python3

from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
from aicsimageio.vendor.omexml import OMEXML
import argparse
import json
import logging
from multiprocessing import Pool
import numpy as np
from os import walk
from pathlib import Path
import re
from shapely.geometry import Polygon
from tifffile import TiffFile
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import yaml

from utils import print_directory_tree

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

TIFF_FILE_NAMING_PATTERN = re.compile( r'^R\d{3}_X(\d{3})_Y(\d{3})\.tif' )


def collect_tiff_file_list(
        directory: Path,
        TIFF_FILE_NAMING_PATTERN: re.Pattern
) -> List[ Path ] :
    """
    Given a directory path and a regex, find all the files in the directory that
    match the regex.

    TODO: this is very similar to a function in create_cellshapes_csv.py -- could
    do to unify with a separate module?
    """
    fileList = []

    for dirpath, dirnames, filenames in walk( directory ) :
        for filename in filenames :
            if TIFF_FILE_NAMING_PATTERN.match( filename ) :
                fileList.append( directory / filename )

    if len( fileList ) == 0 :
        logger.warning( "No files found in " + str( directory ) )

    return fileList


def get_lateral_resolution( cytokit_config_filename: Path ) -> float :
    
    with open(cytokit_config_filename) as cytokit_config_file:
        cytokit_config = yaml.safe_load( cytokit_config_file )

    return float( "%0.2f" % cytokit_config[ "acquisition" ][ "lateral_resolution" ] )


def collect_best_zplanes( dataJsonFile: Path ) -> Dict :
    """
    Given the path to Cytokit's "data.json" file, containing results from the focal
    plane selector, create a dictionary with tile x,y coordinates pointing to the index
    of the best-focus z-plane.
    """
    print('Collecting best z-planes from', dataJsonFile)
    with open( dataJsonFile ) as jsonData:
        dataJsonDict = json.load( jsonData )

    focalPlaneData = dataJsonDict[ "focal_plane_selector" ]

    bestZplanes = {}

    for tileData in focalPlaneData :

        tileX = int( tileData[ "tile_x" ] )
        tileY = int( tileData[ "tile_y" ] )
        bestZ = int( tileData[ "best_z" ] )

        if tileX in bestZplanes :
            bestZplanes[ tileX ][ tileY ] = bestZ
        else :
            bestZplanes[ tileX ] = {
                tileY : bestZ
            }

    return bestZplanes


def collect_expressions_extract_channels( extractFile: Path ) -> List[str]:
    """
    Given a TIFF file path, read file with TiffFile to get Labels attribute from
    ImageJ metadata. Return a list of the channel names in the same order as they
    appear in the ImageJ metadata.
    We need to do this to get the channel names in the correct order, and the
    ImageJ "Labels" attribute isn't picked up by AICSImageIO.
    """
    img = TiffFile( extractFile )

    numChannels = int( img.imagej_metadata[ "channels" ] )

    channelList = img.imagej_metadata[ "Labels" ][ 0:numChannels ]

    # Remove "proc_" from the start of the channel names.
    procPattern = re.compile( r'^proc_(.*)' )

    channelList = [ procPattern.match( channel ).group( 1 ) for channel in channelList ]

    return channelList


def create_roi_polygons(
    imageData: np.ndarray,
    bestZforROI: int,
    omeXml,
) :
    """
    Given a NumPy ndarray of image data, the index of the best focus z-plane, and
    an aicsimageio.vendor.omexml.OMEXML object containing basic OME-XML, create
    strings representing the polygon shapes of segmented cells and add these to the
    "ROI" element of the OME-XML. Return the OME-XML with the ROI element
    populated.
    """
    # TODO: only have cell shapes for now. Probably also want to add nuclei.
    cellBoundaryMask = imageData[ 0, 2, bestZforROI, :, : ]

    omeXmlRoot = ET.fromstring( omeXml.to_xml() )
    roiElement = ET.SubElement(
        omeXmlRoot,
        "ROI",
        attrib = {
            "xmlns" : "http://www.openmicroscopy.org/Schemas/ROI/2016-06",
            "ID" : "ROI:1",
        },
    )
    roiDesc = ET.SubElement( roiElement, "Description" )
    roiDesc.text = "Shapes representing cell boundaries"

    roiUnion = ET.SubElement( roiElement, "Union" )

    for i in range( 1, cellBoundaryMask.max() + 1 ) :
        roiShape = np.where( cellBoundaryMask == i )
        roiShapeTuples = list( zip( roiShape[ 1 ], roiShape[ 0 ] ) )
        polygon = Polygon( roiShapeTuples )
        coords = polygon.exterior.coords

        coordStrings = []

        for coordPair in coords :
            coordPairString = ",".join( [ str( int( c ) ) for c in coordPair ] )
            coordStrings.append( coordPairString )

        allCoordsString = " ".join( coordStrings )

        # Create Polygon with coordinates in.
        roiPolygon = ET.SubElement(
            roiUnion,
            "Polygon",
            attrib = {
                "ID" : "Shape:" + str( i ),
                "Points" : allCoordsString,
                "TheZ" : str( bestZforROI ),
            },
        )

    omeXmlWithROIs = OMEXML( xml = ET.tostring( omeXmlRoot ) )

    return omeXmlWithROIs


def convert_tiff_file(
        funcArgs: Tuple[ Path, Path, List, float, int ]
) :
    """
    Given a tuple containing a source TIFF file path, a destination OME-TIFF path,
    a list of channel names, a float value for the lateral resolution in
    nanometres, and an integer value for the best focus z-plane, convert the
    source TIFF file to OME-TIFF format, containing polygons for segmented cell
    shapes in the "ROI" OME-XML element.
    """
    sourceFile, ometiffFile, channelNames, lateral_resolution, bestZforROI = funcArgs

    logger.info( f"Converting file: { str( sourceFile ) }")

    image = AICSImage( sourceFile )

    imageDataForOmeTiff = image.get_image_data( "TCZYX" )

    # Create a template OME-XML object.
    omeXml = OMEXML()

    # Populate it with image metadata.
    omeXml.image().Pixels.set_SizeT( image.size_t )
    omeXml.image().Pixels.set_SizeC( image.size_c )
    omeXml.image().Pixels.set_SizeZ( image.size_z )
    omeXml.image().Pixels.set_SizeY( image.size_y )
    omeXml.image().Pixels.set_SizeX( image.size_x )
    omeXml.image().Pixels.set_PixelType( str( imageDataForOmeTiff.dtype ) )
    omeXml.image().Pixels.set_DimensionOrder( "XYZCT" )
    omeXml.image().Pixels.channel_count = len( channelNames )
    omeXml.image().Pixels.set_PhysicalSizeX( lateral_resolution )
    #omeXml.image().Pixels.set_PhysicalSizeXUnit( "nm" )
    omeXml.image().Pixels.set_PhysicalSizeY( lateral_resolution )
    #omeXml.image().Pixels.set_PhysicalSizeYUnit( "nm" )

    for i in range( 0, len( channelNames ) ) :
        omeXml.image().Pixels.Channel( i ).Name = channelNames[ i ]
        omeXml.image().Pixels.Channel( i ).ID = "Channel:0:" + str( i )

    # If we've been passed a bestZ, we need to get the ROI info for
    # segmentation mask boundaries, and add it to the OME-XML.
    if bestZforROI is not None :
        omeXml = create_roi_polygons( imageDataForOmeTiff, bestZforROI, omeXml )

    with ome_tiff_writer.OmeTiffWriter( ometiffFile ) as ome_writer :
        ome_writer.save(
            imageDataForOmeTiff,
            ome_xml = omeXml,
            dimension_order = "TCZYX",
            channel_names = channelNames,
        )

    logger.info( f"OME-TIFF file created: { ometiffFile }" )


def find_best_z( sourceFile: Path, bestZplanes: Dict ) -> int :
    """
    Given a tile TIFF file path and a dictionary of best focus z-planes indexed by
    tile x,y coordinates, find the tile's best focus z-plane index and return it.
    """
    filenameMatch = TIFF_FILE_NAMING_PATTERN.match( str( sourceFile.name ) )

    tileX = filenameMatch.group( 1 )
    tileY = filenameMatch.group( 2 )

    bestZ = bestZplanes[ int( tileX ) - 1 ][ int( tileY ) - 1 ]

    return bestZ


def create_ome_tiffs(
        file_list: List[Path],
        output_dir: Path,
        channel_names: List[str],
        lateral_resolution: float,
        subprocesses: int,
        bestZplanes: Dict = None
):
    """
    Given:
        - a list of TIFF files
        - an output directory path
        - a list of channel names
        - a float value for the lateral resolution in nanometres (aka XY resolution aka pixel size).
        - an integer value for the number of multiprocessing subprocesses
        - a dictionary of best focus z-planes indexed by tile x,y coordinates
    Create OME-TIFF files using parallel processes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    args_for_conversion = []

    for source_file in file_list:
        ome_tiff_file = (output_dir / source_file.name).with_suffix(".ome.tiff")

        bestZforROI = None
        if bestZplanes :
            bestZforROI = find_best_z( source_file, bestZplanes )

        args_for_conversion.append(
            (
                source_file,
                ome_tiff_file,
                channel_names,
                lateral_resolution,
                bestZforROI,
            )
        )

    for argtuple in args_for_conversion :
        convert_tiff_file( argtuple )

    #with Pool(processes=subprocesses) as pool:
    #    pool.imap_unordered(convert_tiff_file, args_for_conversion)
    #    pool.close()
    #    pool.join()


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
        "cytokit_processor_output",
        help="Path to output of `cytokit processor`",
        type=Path,
    )
    parser.add_argument(
        "cytokit_operator_output",
        help="Path to output of `cytokit operator`",
        type=Path,
    )
    parser.add_argument(
        "cytokit_config",
        help="Path to Cytokit YAML config file",
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

    print('Cytokit processor output:')
    print_directory_tree(args.cytokit_processor_output)
    print('Cytokit operator output:')
    print_directory_tree(args.cytokit_operator_output)

    output_dir = Path( 'output' )
    output_dir.mkdir(parents=True, exist_ok=True)

    cytometry_tile_dir_piece = Path("cytometry/tile")
    extract_expressions_piece = Path("extract/expressions")
    processor_data_json_piece = Path("processor/data.json")

    cytometryTileDir = args.cytokit_processor_output / cytometry_tile_dir_piece
    print('Cytometry tile directory:', cytometryTileDir)
    extractDir = args.cytokit_operator_output / extract_expressions_piece
    print('Extract expressions directory:', extractDir)

    segmentationFileList = collect_tiff_file_list( cytometryTileDir, TIFF_FILE_NAMING_PATTERN )
    extractFileList = collect_tiff_file_list( extractDir, TIFF_FILE_NAMING_PATTERN )
    
    lateral_resolution = get_lateral_resolution( args.cytokit_config )

    # For each tile, find the best focus Z plane.
    bestZplanes = collect_best_zplanes( args.cytokit_processor_output / processor_data_json_piece )

    # Create segmentation mask OME-TIFFs
    if segmentationFileList:
        create_ome_tiffs(
            segmentationFileList,
            output_dir / cytometry_tile_dir_piece / 'ome-tiff',
            SEGMENTATION_CHANNEL_NAMES,
            lateral_resolution,
            args.processes,
            bestZplanes,
        )

    # Create the extract OME-TIFFs.
    if extractFileList:
        # For the extract, pull the correctly ordered list of channel names from
        # one of the files, as they aren't guaranteed to be in the same order as
        # the YAML config.
        extractChannelNames = collect_expressions_extract_channels( extractFileList[ 0 ] )

        create_ome_tiffs(
            extractFileList,
            output_dir / extract_expressions_piece / 'ome-tiff',
            extractChannelNames,
            lateral_resolution,
            args.processes,
        )
