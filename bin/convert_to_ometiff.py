#!/usr/bin/env python3

import argparse
import json
import logging
import re
from multiprocessing import Pool
from os import walk
from pathlib import Path
from typing import List, Tuple

import lxml.etree
import numpy as np
import yaml
from aicsimageio import AICSImage
from aicsimageio.vendor.omexml import OMEXML
from aicsimageio.writers import ome_tiff_writer
from tifffile import TiffFile

from utils import print_directory_tree

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)


SEGMENTATION_CHANNEL_NAMES = [
    "cells",
    "nuclei",
    "cell_boundaries",
    "nucleus_boundaries",
]

TIFF_FILE_NAMING_PATTERN = re.compile(r"^R\d{3}_X(\d{3})_Y(\d{3})\.tif")


def collect_tiff_file_list(directory: Path, TIFF_FILE_NAMING_PATTERN: re.Pattern) -> List[Path]:
    """
    Given a directory path and a regex, find all the files in the directory that
    match the regex.

    TODO: this is very similar to a function in create_cellshapes_csv.py -- could
    do to unify with a separate module?
    """
    fileList = []

    for dirpath, dirnames, filenames in walk(directory):
        for filename in filenames:
            if TIFF_FILE_NAMING_PATTERN.match(filename):
                fileList.append(directory / filename)

    if len(fileList) == 0:
        logger.warning("No files found in " + str(directory))

    return fileList


def get_lateral_resolution(cytokit_config_filename: Path) -> float:

    with open(cytokit_config_filename) as cytokit_config_file:
        cytokit_config = yaml.safe_load(cytokit_config_file)

    return float("%0.2f" % cytokit_config["acquisition"]["lateral_resolution"])


def collect_expressions_extract_channels(extractFile: Path) -> List[str]:
    """
    Given a TIFF file path, read file with TiffFile to get Labels attribute from
    ImageJ metadata. Return a list of the channel names in the same order as they
    appear in the ImageJ metadata.
    We need to do this to get the channel names in the correct order, and the
    ImageJ "Labels" attribute isn't picked up by AICSImageIO.
    """
    img = TiffFile(extractFile)

    numChannels = int(img.imagej_metadata["channels"])

    channelList = img.imagej_metadata["Labels"][0:numChannels]

    # Remove "proc_" from the start of the channel names.
    procPattern = re.compile(r"^proc_(.*)")

    channelList = [procPattern.match(channel).group(1) for channel in channelList]

    return channelList


def add_pixel_size_units(omeXml):
    # Don't take any chances about locale environment variables in Docker containers
    # and headless server systems; be explicit about using UTF-8
    encoding = "utf-8"
    omeXmlRoot = lxml.etree.fromstring(omeXml.to_xml(encoding=encoding).encode(encoding))

    namespace_prefix = omeXmlRoot.nsmap[None]
    image_node = omeXmlRoot.find(f"{{{namespace_prefix}}}Image")
    pixels_node = image_node.find(f"{{{namespace_prefix}}}Pixels")

    pixels_node.set("PhysicalSizeXUnit", "nm")
    pixels_node.set("PhysicalSizeYUnit", "nm")

    omexml_with_pixel_units = OMEXML(xml=lxml.etree.tostring(omeXmlRoot))
    return omexml_with_pixel_units


def convert_tiff_file(funcArgs: Tuple[Path, Path, List, float]):
    """
    Given a tuple containing a source TIFF file path, a destination OME-TIFF path,
    a list of channel names, a float value for the lateral resolution in
    nanometres, convert the source TIFF file to OME-TIFF format, containing
    polygons for segmented cell shapes in the "ROI" OME-XML element.
    """
    sourceFile, ometiffFile, channelNames, lateral_resolution = funcArgs

    logger.info(f"Converting file: { str( sourceFile ) }")

    image = AICSImage(sourceFile)

    imageDataForOmeTiff = image.get_image_data("TCZYX")

    # Create a template OME-XML object.
    omeXml = OMEXML()

    # Populate it with image metadata.
    omeXml.image().Pixels.set_SizeT(image.size_t)
    omeXml.image().Pixels.set_SizeC(image.size_c)
    omeXml.image().Pixels.set_SizeZ(image.size_z)
    omeXml.image().Pixels.set_SizeY(image.size_y)
    omeXml.image().Pixels.set_SizeX(image.size_x)
    omeXml.image().Pixels.set_PixelType(str(imageDataForOmeTiff.dtype))
    omeXml.image().Pixels.set_DimensionOrder("XYZCT")
    omeXml.image().Pixels.channel_count = len(channelNames)
    omeXml.image().Pixels.set_PhysicalSizeX(lateral_resolution)
    omeXml.image().Pixels.set_PhysicalSizeY(lateral_resolution)

    omeXml = add_pixel_size_units(omeXml)

    for i in range(0, len(channelNames)):
        omeXml.image().Pixels.Channel(i).Name = channelNames[i]
        omeXml.image().Pixels.Channel(i).ID = "Channel:0:" + str(i)

    with ome_tiff_writer.OmeTiffWriter(ometiffFile) as ome_writer:
        ome_writer.save(
            imageDataForOmeTiff,
            ome_xml=omeXml,
            dimension_order="TCZYX",
            channel_names=channelNames,
        )

    logger.info(f"OME-TIFF file created: { ometiffFile }")


def create_ome_tiffs(
    file_list: List[Path],
    output_dir: Path,
    channel_names: List[str],
    lateral_resolution: float,
    subprocesses: int,
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

        args_for_conversion.append(
            (
                source_file,
                ome_tiff_file,
                channel_names,
                lateral_resolution,
            )
        )

    # for argtuple in args_for_conversion :
    #    convert_tiff_file( argtuple )

    with Pool(processes=subprocesses) as pool:
        pool.imap_unordered(convert_tiff_file, args_for_conversion)
        pool.close()
        pool.join()


########
# MAIN #
########
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Convert Cytokit's output TIFFs containing segmentation and extraction "
            'results to OME-TIFF, and add the channel names. Creates an "ome-tiff" '
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
        "-p",
        "--processes",
        help="Number of parallel OME-TIFF conversions to perform at once",
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

    print("Cytokit processor output:")
    print_directory_tree(args.cytokit_processor_output)
    print("Cytokit operator output:")
    print_directory_tree(args.cytokit_operator_output)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    cytometry_tile_dir_piece = Path("cytometry/tile")
    extract_expressions_piece = Path("extract/expressions")
    processor_data_json_piece = Path("processor/data.json")

    cytometryTileDir = args.cytokit_processor_output / cytometry_tile_dir_piece
    print("Cytometry tile directory:", cytometryTileDir)
    extractDir = args.cytokit_operator_output / extract_expressions_piece
    print("Extract expressions directory:", extractDir)

    segmentationFileList = collect_tiff_file_list(cytometryTileDir, TIFF_FILE_NAMING_PATTERN)
    extractFileList = collect_tiff_file_list(extractDir, TIFF_FILE_NAMING_PATTERN)

    lateral_resolution = get_lateral_resolution(args.cytokit_config)

    # Create segmentation mask OME-TIFFs
    if segmentationFileList:
        create_ome_tiffs(
            segmentationFileList,
            output_dir / cytometry_tile_dir_piece / "ome-tiff",
            SEGMENTATION_CHANNEL_NAMES,
            lateral_resolution,
            args.processes,
        )

    # Create the extract OME-TIFFs.
    if extractFileList:
        # For the extract, pull the correctly ordered list of channel names from
        # one of the files, as they aren't guaranteed to be in the same order as
        # the YAML config.
        extractChannelNames = collect_expressions_extract_channels(extractFileList[0])

        create_ome_tiffs(
            extractFileList,
            output_dir / extract_expressions_piece / "ome-tiff",
            extractChannelNames,
            lateral_resolution,
            args.processes,
        )
