#!/usr/bin/env python3

import argparse
import json
import logging
import os
from pathlib import Path
import re
import stat
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patterns for detecting raw data files are below.
# We follow Cytokit's "keyence_multi_cycle_v01" naming convention defined in:
# https://github.com/hammerlab/cytokit/blob/master/python/pipeline/cytokit/io.py
# Pattern for the directories containing the raw data from each cycle-region
# pair. Different submitters use different naming conventions (e.g.
# cyc001_reg001_191209_123455 or Cyc1_reg1), so our regex has to allow for this.
rawDirNamingPattern = re.compile( r'^cyc0*(\d+)_reg0*(\d+).*', re.IGNORECASE )
# Pattern for raw data TIFF files. These should be named according to the following pattern:
# <region index>_<tile index>_Z<z-plane index>_CH<channel index>.tif
# All indices start at 1.
# Tile index is padded to three digits, e.g. 00025, 00001, etc.
# Z-plane index is padded to three digits, e.g. 025, 001, etc.
# Region and channel indices are one digit each.
rawFileNamingPattern = re.compile( r'^\d_\d{5}_Z\d{3}_CH\d\.tif$' )
# Pattern to match one single digit at the start of a string, used to replace
# incorrect region indices with the correct ones in some raw data TIFF files.
rawFileRegionPattern = re.compile( r'^\d' )


def main(data_dir: str):
    ###################################################################
    # Inspect source directories and collect paths to raw data files. #
    ###################################################################

    # Ensure that source directory exists and is readable.
    st = os.stat( data_dir )
    readable = bool( st.st_mode & stat.S_IRUSR )
    if not readable :
        logger.error(
            "Source directory " +
            data_dir +
            " is not readable by the current user."
        )
        sys.exit(1)


    # Get list of contents of source directory. This should contain a set of
    # subdirectories, one for each cycle-region pair.
    sourceDirList = None
    try :
        sourceDirList = os.listdir( data_dir )
    except OSError as err :
        logger.error(
            "Could not acquire list of contents for " +
            data_dir +
            " : " +
            err.strerror
        )
        sys.exit(1)

    # Filter the contents list of the source directory for directories matching
    # the expected raw data directory naming pattern (cycle-region pairs).
    # Different submitters follow different naming conventions currently.
    sourceDataDirs = list(
        filter(
            rawDirNamingPattern.search,
            sourceDirList
        )
    )
    # If there were no matching directories found, exit.
    if len( sourceDataDirs ) == 0 :

        logger.error(
            "No directories matching expected raw data directory naming pattern found in " +
            data_dir
        )
        sys.exit(1)


    # Go through the cycle-region directories and get a list of the contents of
    # each one. Each cycle-region directory should contain TIFF files,
    # following the raw data file naming convention defined above.
    # Collect raw data file names in a dictionary, indexed by directory name.
    sourceDataFiles = {}
    for sdir in sourceDataDirs :

        fileList = None

        try :
            fileList = os.listdir( os.path.join( data_dir, sdir ) )
        except OSError as err :
            logger.error(
                "Could not acquire list of contents for " +
                sdir +
                " : " +
                err.strerror
            )
            sys.exit(1)

        # Validate naming pattern of raw data files according to pattern
        # defined above.
        fileList = list(
            filter(
                rawFileNamingPattern.search,
                fileList
            )
        )

        # Die if we didn't get any matching files.
        if len( fileList ) == 0 :
            logger.error(
                "No files found matching expected raw file naming pattern in " +
                sdir
            )
            sys.exit(1)

        # Otherwise, collect the list of matching file names in the dictionary.
        else :
            sourceDataFiles[ sdir ] = fileList


    # Check that expected source data files are all present. We know, from the
    # pipeline config, the number of regions, cycles, z-planes, and channels, so we
    # should be able to verify that we have one file per channel, per z-plane,
    # per cycle, per region.

    # Since the files will have had to match the required naming pattern, we
    # know that they'll be named basically as expected. A simple check would be
    # to just count the number of files present and see if we have the expected
    # number for each region, cycle, and z-plane.

    # For each region, we should have num_cycles * (region_height * region_width ) * num_z_planes * len( per_cycle_channel_names ) files.
    # If we do, we could stop there? It's not a super rigorous check but we already know we have files named correctly...

    # If we don't, we can inspect each cycle. For each cycle, we should have ...



    ######################################
    # Start creating directories and links
    ######################################

    targetDirectory = "symlinks"

    # Create target directory.
    try :
        os.mkdir("symlinks")
    except OSError as err :
        logger.error(
            "Could not create Cytokit data directory " +
            targetDirectory +
            " : " +
            err.strerror
        )
        sys.exit(1)
    else :
        logger.info( "Cytokit data directory created at %s" % targetDirectory )

    for sdir in sourceDataFiles :

        dirMatch = rawDirNamingPattern.match( sdir )

        cycle, region = dirMatch.group( 1, 2 )

        cycleRegionDir = os.path.join( "symlinks", "Cyc" + cycle + "_reg" + region )

        try :
            os.mkdir( cycleRegionDir )
        except OSError as err :
            logger.error(
                "Could not create target directory " +
                cycleRegionDir +
                " : " +
                err.strerror
            )
            sys.exit(1)

        # Create symlinks for TIFF files.
        for tifFileName in sourceDataFiles[ sdir ] :

            # Replace the region number at the start because sometimes it's wrong.
            linkTifFileName = rawFileRegionPattern.sub( region, tifFileName )

            # Set up full path to symlink.
            linkTifFilePath = os.path.join( cycleRegionDir, linkTifFileName )

            # Full path to source raw data file.
            sourceTifFilePath = os.path.join( data_dir, sdir, tifFileName )

            # Create the symlink.
            try :
                print('Linking', sourceTifFilePath, 'to', linkTifFilePath)
                os.symlink(
                    sourceTifFilePath,
                    linkTifFilePath
                )
            except OSError as err :
                logger.error(
                    "Count not create symbolic link: " +
                    err.strerror
                )
                sys.exit(1)

    logger.info( "Links created in directories under %s" % targetDirectory )


########
# MAIN #
########
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = "Create a directory and populate directory with directories containing symlinks to the raw image data."
    )
    parser.add_argument(
        "data_dir",
        help="Data directory",
    )

    args = parser.parse_args()

    main(args.data_dir)
