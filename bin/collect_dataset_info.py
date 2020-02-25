#!/usr/bin/env python3

# This script is not part of the automatic pipeline. It should be run manually
# and is hopefully just a temporary necessity until we have standardised
# submission formats.

import argparse
from collections import Counter
import datetime
import json
import logging
import math
import re
import sys
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)


# collect_attribute()
# Returns the contents of the field matching the name(s) passed in the
# fieldNames argument.
# Field names are passed in a list because sometimes there is more than one
# possible name for the field. For example, in some files, the date field is
# called "date" and in others it is called "dateProcessed".
# The order the field names are passed in this list matters, because only the
# first one to match is returned.
# In some cases, the config file contains both versions of the field name, but
# only one of them has valid content. For example, some files have:
#   - "aperture": 0.75
#   - "numerical_aperture": 0.0
# So in this case we would only want the contents of the "aperture" field.
# In other cases, the "aperture" field doesn't exist, and only the
# "numerical_aperture" field is present, with valid content.
def collect_attribute( fieldNames, configDict: Dict ) :

    for fieldName in fieldNames:
        if fieldName in configDict:
            return configDict[ fieldName ]

    # If we're still here, it means we tried all the possible field names and
    # didn't find a match in the config, so we have to fail.
    fieldNameString = ", ".join( fieldNames )
    logger.error( "No match found for field name(s) in config: %s" % fieldNameString )
    sys.exit(1)


def infer_channel_name_from_index(
        cycleIndex: int,
        channelIndex: int,
        channelNames,
        channelsPerCycle
):

    # If there is no cycle+channel set for a particular measurement, then the
    # cycle (or channel?) index is set to "-1". E.g. if no membrane stain
    # channel exists, the membraneStainCycle can be set to "-1". Just return
    # None in this case.
    if any( x == -1 for x in [ cycleIndex, channelIndex ] ) :
        return None

    cycleLastChannelIdx = cycleIndex * channelsPerCycle

    cycleChannelIndices = range( cycleLastChannelIdx - channelsPerCycle, cycleLastChannelIdx )

    channelNameIdx = cycleChannelIndices[ channelIndex - 1 ]

    return channelNames[ channelNameIdx ]


# calculate_target_shape()
# Cytokit's nuclei detection U-Net (from CellProfiler) works best at 20x magnification.
# The CellProfiler U-Net requires the height and width of the images to be
# evenly divisible by 2 raised to the number of layers in the network, in this case 2^3=8.
# https://github.com/hammerlab/cytokit/issues/14
# https://github.com/CellProfiler/CellProfiler-plugins/issues/65
def calculate_target_shape( magnification: int, tileHeight: int, tileWidth: int ) :
    scaleFactor = 1
    if magnification != 20 :
        scaleFactor = 20 / magnification

    dims = {
        "height" : tileHeight,
        "width" : tileWidth,
    }

    # Width and height must be evenly divisible by 8, so we round them up to them
    # closest factor of 8 if they aren't.
    for dimension in dims:
        if dims[ dimension ] % 8 :
            newDim = int( 8 * math.ceil( float( dims[ dimension ] )/8 ) )
            dims[ dimension ] = newDim

    return [ dims[ "height" ], dims[ "width" ] ]


# make_channel_names_unique( channelNames )
# Sometimes channel names are not unique, e.g. if DAPI was used in every cycle,
# sometimes each DAPI channel is just named "DAPI", other times they are named
# "DAPI1", "DAPI2", "DAPI3", etc. The latter is better, because it enables us
# to select the specific DAPI channel from the correct cycle to use for
# segmentation and/or for best focus plane selection. So, if there are
# duplicated channel names, we will append an index, starting at 1, to each
# occurrence of the channel name.
def make_channel_names_unique( channelNames ) :

    uniqueNames = Counter(channelNames)

    newNames = []

    seenCounts = {}

    for channel in channelNames :
        if uniqueNames[ channel ] > 1 :
            if channel in seenCounts :
                newNames.append( channel + "_" + str( seenCounts[ channel ] + 1 ) )
                seenCounts[ channel ] += 1
            else :
                newNames.append( channel + "_1" )
                seenCounts[ channel ] = 1
        else :
            newNames.append( channel )

    return newNames



########
# MAIN #
########
if __name__ == "__main__" :
    # Set up argument parser and parse the command line arguments.
    parser = argparse.ArgumentParser(
        description = "Collect information required to perform analysis of a CODEX dataset, from various sources depending on submitted files. This script should be run manually after inspection of submission directories, and is hopefully only a temporary necessity until submission formats have been standardised."
    )
    parser.add_argument(
        "hubmapDatasetID",
        help = "HuBMAP dataset ID, e.g. HBM123.ABCD.456."
    )
    parser.add_argument(
        "rawDataLocation",
        help = "Path to directory containing raw data subdirectories (named with cycle and region numbers)."
    )
    parser.add_argument(
        "exptJsonFileName",
        help = "Path to experiment.json file from CODEX Toolkit pipeline."
    )
    parser.add_argument(
        "--segm-json",
        help = "Path to JSON file containing segmentation parameters (including nuclearStainChannel and nuclearStainCycle)."
    )
    parser.add_argument(
        "--segm-text",
        help = "Path to text file containing segmentation parameters (including nuclearStainChannel and nuclearStainCycle). This is usually found in the \"processed\" directory in the submitted data."
    )
    parser.add_argument(
        "-c",
        "--channel-names",
        help = "Path to text file containing list of channel names, if necessary."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help = "Path to output file pipeline config (JSON format). Default: ./<dataset ID>_pipelineConfig.json."
    )

    args = parser.parse_args()


    if not args.segm_json and not args.segm_text :
        logger.error( "Segmentation parameters file name not provided. Cannot continue." )
        sys.exit(1)

    if args.segm_json and args.segm_text :
        logger.warning(
            "Segmentation parameter files " +
            args.segm_json +
            " and " +
            args.segm_text +
            " provided. Will only use " +
            args.segm_json
        )

    if not args.outfile :
        args.outfile = args.hubmapDatasetID + "_pipelineConfig.json"

    if not args.channel_names :
        logger.info( "No channel names file passed. Will look for channel names in experiment JSON config." )

    logger.info( "Reading config from " + args.exptJsonFileName + "..." )

    # Read in the experiment JSON config.
    with open( args.exptJsonFileName, 'r' ) as exptJsonFile :
        exptJsonData = exptJsonFile.read()
    logger.info( "Finished reading file " + args.exptJsonFileName )

    # Create dictionary from experiment JSON config.
    exptConfigDict = json.loads( exptJsonData )

    # Read in the segmentation parameters. If we have a JSON file, use that.
    if args.segm_json :
        logger.info( "Reading segmentation parameters from " + args.segm_json + "..." )
        with open( args.segm_json, 'r' ) as segmJsonFile :
            segmJsonData = segmJsonFile.read()
        segmParams = json.loads( segmJsonData )
    else :

        logger.info( "Reading segmentation parameters from " + args.segm_text + "..." )
        with open( args.segm_text, 'r' ) as segmTextFile :
            fileLines = segmTextFile.read().splitlines()
            segmParams = {}
            for line in fileLines :
                fieldName, fieldContents = line.split( "=" )
                numPattern = re.compile( "^[0-9]+$" )
                numMatch = numPattern.match( fieldContents )
                if numMatch :
                    fieldContents = int( fieldContents )
                segmParams[ fieldName ] = fieldContents

    logger.info( "Finished reading segmentation parameters." )


    datasetInfo = {}

    datasetInfo[ "name" ] = args.hubmapDatasetID
    datasetInfo[ "date" ] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datasetInfo[ "raw_data_location" ] = args.rawDataLocation


    datasetInfo[ "emission_wavelengths" ] = collect_attribute( [ "emission_wavelengths", "wavelengths" ], exptConfigDict )
    datasetInfo[ "axial_resolution" ] = collect_attribute( [ "zPitch", "z_pitch" ], exptConfigDict )
    datasetInfo[ "lateral_resolution" ] = collect_attribute( [ "xyResolution", "per_pixel_XY_resolution" ], exptConfigDict )
    datasetInfo[ "magnification" ] = collect_attribute( [ "magnification" ], exptConfigDict )
    datasetInfo[ "num_z_planes" ] = collect_attribute( [ "num_z_planes" ], exptConfigDict )
    datasetInfo[ "numerical_aperture" ] = collect_attribute( [ "aperture", "numerical_aperture" ], exptConfigDict )
    datasetInfo[ "objective_type" ] = collect_attribute( [ "objectiveType" ], exptConfigDict )
    datasetInfo[ "region_names" ] = collect_attribute( [ "region_names" ], exptConfigDict )
    datasetInfo[ "region_height" ] = collect_attribute( [ "region_height" ], exptConfigDict )
    datasetInfo[ "region_width" ] = collect_attribute( [ "region_width" ], exptConfigDict )
    datasetInfo[ "tile_height" ] = collect_attribute( [ "tile_height" ], exptConfigDict )
    datasetInfo[ "tile_width" ] = collect_attribute( [ "tile_width" ], exptConfigDict )
    datasetInfo[ "tile_overlap_x" ] = collect_attribute( [ "tile_overlap_X" ], exptConfigDict )
    datasetInfo[ "tile_overlap_y" ] = collect_attribute( [ "tile_overlap_Y" ], exptConfigDict )
    datasetInfo[ "tiling_mode" ] = collect_attribute( [ "tiling_mode" ], exptConfigDict )
    datasetInfo[ "per_cycle_channel_names" ] = collect_attribute( [ "channel_names" ], exptConfigDict )
    # Collect channel names.
    channelNames = None

    if args.channel_names :
        with open( args.channel_names, 'r' ) as channelNamesFile :
            channelNames = channelNamesFile.read().splitlines()
    elif "channelNames" in exptConfigDict :
        channelNames = collect_attribute( [ "channelNamesArray" ], exptConfigDict[ "channelNames" ] )
    else :
        logger.error( "Cannot find data for channel_names field." )
        sys.exit(1)

    # If there are identical channel names, make them unique by adding
    # incremental numbers to the end.
    channelNames = make_channel_names_unique( channelNames )

    datasetInfo[ "channel_names" ] = channelNames

    datasetInfo[ "num_cycles" ] = int(
        len( channelNames ) / len( datasetInfo[ "per_cycle_channel_names" ] )
    )

    bestFocusChannel = collect_attribute( [ "bestFocusReferenceChannel", "best_focus_channel" ], exptConfigDict )
    bestFocusCycle = collect_attribute( [ "bestFocusReferenceCycle" ], exptConfigDict )
    bestFocusChannelName = infer_channel_name_from_index(
        int( bestFocusCycle ),
        int( bestFocusChannel ),
        datasetInfo[ "channel_names" ],
        len( datasetInfo[ "per_cycle_channel_names" ] )
    )

    driftCompChannel = collect_attribute( [ "driftCompReferenceChannel", "drift_comp_channel" ], exptConfigDict )
    driftCompCycle = collect_attribute( [ "driftCompReferenceCycle" ], exptConfigDict )
    driftCompChannelName = infer_channel_name_from_index(
        int( driftCompCycle ),
        int( driftCompChannel ),
        datasetInfo[ "channel_names" ],
        len( datasetInfo[ "per_cycle_channel_names" ] )
    )

    datasetInfo[ "best_focus" ] = bestFocusChannelName
    datasetInfo[ "drift_compensation" ] = driftCompChannelName

    nucleiChannel = collect_attribute( [ "nuclearStainChannel" ], segmParams )
    nucleiCycle = collect_attribute( [ "nuclearStainCycle" ], segmParams )
    nucleiChannelName = infer_channel_name_from_index(
        int( nucleiCycle ),
        int( nucleiChannel ),
        datasetInfo[ "channel_names" ],
        len( datasetInfo[ "per_cycle_channel_names" ] )
    )

    # If we don't have a nuclei channel, we can't continue.
    if nucleiChannelName is None :
        logger.error( "No nuclei stain channel found. Cannot continue." )
        sys.exit( 1 )

    membraneChannel = collect_attribute( [ "membraneStainChannel" ], segmParams )
    membraneCycle = collect_attribute( [ "membraneStainCycle" ], segmParams )
    membraneChannelName = infer_channel_name_from_index(
        int( membraneCycle ),
        int( membraneChannel ),
        datasetInfo[ "channel_names" ],
        len( datasetInfo[ "per_cycle_channel_names" ] )
    )

    datasetInfo[ "nuclei_channel" ] = nucleiChannelName

    if membraneChannelName is not None :
        datasetInfo[ "membrane_channel" ] = membraneChannelName


    # The target_shape needs to be worked out based on the metadata. See
    # comments on calculate_target_shape() function definition.
    datasetInfo[ "target_shape" ] = calculate_target_shape(
        datasetInfo[ "magnification" ],
        datasetInfo[ "tile_height" ],
        datasetInfo[ "tile_width" ],
    )


    ##############################
    # Write JSON pipeline config #
    ##############################
    logger.info( "Writing pipeline config..." )
    with open( args.outfile, 'w' ) as outfile:
        json.dump( datasetInfo, outfile, indent = 4 )

    logger.info( "Written pipeline config to " + args.outfile )
