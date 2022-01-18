#!/usr/bin/env python3
import argparse
import csv
import datetime
import json
import logging
import math
import re
from collections import Counter, defaultdict
from os import fspath, walk
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from pipeline_utils.dataset_listing import get_tile_dtype, get_tile_shape

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)


NONRAW_DIRECTORY_NAME_PIECES = [
    "processed",
    "drv",
    "metadata",
    "extras",
    "Overview",
]


# TODO: don't duplicate this in ../cytokit-docker/setup_data_directory.py.
#   Can't share code easily because these files go to different containers
RAW_DIR_NAMING_PATTERN = re.compile(r"^cyc(\d+)_(?P<region>reg\d+).*", re.IGNORECASE)

# Checked with .match below, but be extra explicit that we care about
# the entire string being composed of \x00 characters
ALL_ZERO_CHANNEL_NAME_PATTERN = re.compile("^\x00+$")


def find_files(
    base_directory: Path,
    filename: str,
    ignore_processed_derived_metadata_dirs: bool = False,
) -> List[Path]:
    """
    This returns a full list instead of a generator function because we very
    frequently care about the number of files returned, and it's a waste
    to require most usages of this to be wrapped in `list()`.

    :param base_directory:
    :param filename:
    :param ignore_processed_derived_metadata_dirs:
    :return:
    """
    file_paths = []

    for dirpath, dirnames, filenames in walk(base_directory):
        if ignore_processed_derived_metadata_dirs:
            # Skip any directory that has 'processed' in the name.
            # Since deleting items from a Python list takes linear time, be a little
            # fancier: find directory names that we *should* recurse into, clear the
            # list, and then re-populate. Probably not necessary when top-level
            # directories only have 4-5 children, but quadratic runtime feels wrong
            dirnames_to_recurse = []
            for dirname in dirnames:
                if not any(piece in dirname for piece in NONRAW_DIRECTORY_NAME_PIECES):
                    dirnames_to_recurse.append(dirname)
            dirnames.clear()
            dirnames.extend(dirnames_to_recurse)

        # case-insensitive match
        cf_filename_mapping = {fn.casefold(): fn for fn in filenames}
        casefolded_filename = filename.casefold()
        if casefolded_filename in cf_filename_mapping:
            file_paths.append(base_directory / dirpath / cf_filename_mapping[casefolded_filename])

    return file_paths


def collect_attribute(fieldNames: List[str], configDict: Dict):
    """
    Returns the contents of the field matching the name(s) passed in the
    fieldNames argument.
    Field names are passed in a list because sometimes there is more than one
    possible name for the field. For example, in some files, the date field is
    called "date" and in others it is called "dateProcessed".
    The order the field names are passed in this list matters, because only the
    first one to match is returned.
    In some cases, the config file contains both versions of the field name, but
    only one of them has valid content. For example, some files have:
      - "aperture": 0.75
      - "numerical_aperture": 0.0
    So in this case we would only want the contents of the "aperture" field.
    In other cases, the "aperture" field doesn't exist, and only the
    "numerical_aperture" field is present, with valid content.
    """

    for fieldName in fieldNames:
        if fieldName in configDict:
            return configDict[fieldName]

    # If we're still here, it means we tried all the possible field names and
    # didn't find a match in the config, so we have to fail.
    fieldNameString = ", ".join(fieldNames)
    raise KeyError(f"No match found for field name(s) in config: {fieldNameString}")


def infer_channel_name_from_index(
    cycleIndex: int,
    channelIndex: int,
    channelNames: List[str],
    channelsPerCycle: int,
) -> Optional[str]:

    # If there is no cycle+channel set for a particular measurement, then the
    # cycle (or channel?) index is set to "-1". E.g. if no membrane stain
    # channel exists, the membraneStainCycle can be set to "-1". Just return
    # None in this case.
    if -1 in {cycleIndex, channelIndex}:
        return None

    cycleLastChannelIdx = cycleIndex * channelsPerCycle

    cycleChannelIndices = range(cycleLastChannelIdx - channelsPerCycle, cycleLastChannelIdx)

    channelNameIdx = cycleChannelIndices[channelIndex - 1]

    return channelNames[channelNameIdx]


def calculate_target_shape(magnification: int, tileHeight: int, tileWidth: int):
    """
    Cytokit's nuclei detection U-Net (from CellProfiler) works best at 20x magnification.
    The CellProfiler U-Net requires the height and width of the images to be
    evenly divisible by 2 raised to the number of layers in the network, in this case 2^3=8.
    https://github.com/hammerlab/cytokit/issues/14
    https://github.com/CellProfiler/CellProfiler-plugins/issues/65
    """
    scaleFactor = 1
    if magnification != 20:
        scaleFactor = 20 / magnification

    dims = {
        "height": tileHeight,
        "width": tileWidth,
    }

    # Width and height must be evenly divisible by 8, so we round them up to them
    # closest factor of 8 if they aren't.
    for dimension in dims:
        if dims[dimension] % 8:
            newDim = int(8 * math.ceil(float(dims[dimension]) / 8))
            dims[dimension] = newDim

    return [dims["height"], dims["width"]]


def make_DAPI_channel_names_unique(channelNames: List[str]) -> List[str]:
    """
    Sometimes DAPI channel names are not unique, e.g. if DAPI was used in every
    cycle, sometimes each DAPI channel is just named "DAPI", other times they
    are named "DAPI1", "DAPI2", "DAPI3", etc. The latter is better, because it
    enables us to select the specific DAPI channel from the correct cycle to
    use for segmentation and/or for best focus plane selection. So, if there
    are duplicated channel names, we will append an index, starting at 1, to
    each occurrence of the channel name.
    """

    uniqueNames = Counter(channelNames)

    newNames = []

    dapiCount = 1

    for channel in channelNames:
        if channel == "DAPI":
            if uniqueNames[channel] > 1:
                newNames.append(f"{channel}_{dapiCount}")
                dapiCount += 1
            else:
                newNames.append(channel)
        else:
            newNames.append(channel)
    return newNames


def warn_if_multiple_files(paths: List[Path], label: str):
    if len(paths) > 1:
        message_pieces = [f"Found multiple {label} files:"]
        message_pieces.extend(f"\t{path}" for path in paths)
        logger.warning("\n".join(message_pieces))
        # TODO: throw an exception?


def highest_file_sort_key(seg_text_file: Path) -> Tuple[int, int]:
    """
    Get the highest-level file (shallowest in a directory tree) among multiple
    files with the same name. If equal, fall back to length of the full file
    path (since the presence of a "processed" directory will make the path longer).
    :param seg_text_file:
    :return:
    """
    return len(seg_text_file.parts), len(fspath(seg_text_file))


def find_raw_data_dir(directory: Path) -> Path:
    raw_data_dir_possibilities = []

    for child in directory.iterdir():
        if not child.is_dir():
            continue
        if not any(piece in child.name for piece in NONRAW_DIRECTORY_NAME_PIECES):
            raw_data_dir_possibilities.append(child)

    if len(raw_data_dir_possibilities) > 1:
        message_pieces = ["Found multiple raw data directory possibilities:"]
        message_pieces.extend(f"\t{path}" for path in raw_data_dir_possibilities)
        raise ValueError("\n".join(message_pieces))

    return raw_data_dir_possibilities[0]


def get_region_names_from_directories(base_path: Path) -> List[str]:
    regions = set()
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        else:
            m = RAW_DIR_NAMING_PATTERN.match(child.name)
            if m:
                regions.add(m.group("region"))
    return sorted(regions)


def calculate_pixel_overlaps_from_proportional(target_key: str, exptConfigDict: Dict) -> int:

    if target_key != "tile_overlap_x" and target_key != "tile_overlap_y":
        raise ValueError(f"Invalid target_key for looking up tile overlap: {target_key}")

    components = target_key.split("_")

    overlap_proportion_key = components[0] + "".join(x.title() for x in components[1:])

    overlap_proportion = collect_attribute([overlap_proportion_key], exptConfigDict)

    # Fail if we find something >1 here, proportions can't be >1.
    if overlap_proportion > 1:
        raise ValueError(
            f"Tile overlap proportion at key {overlap_proportion_key} is greater than 1; this doesn't make sense."
        )

    # If we're still here then we need to get the size of the appropriate
    # dimension in pixels, so that we can calculate the overlap in pixels.
    dimension_mapping = {"x": "tileWidth", "y": "tileHeight"}
    target_dimension = dimension_mapping[components[2]]

    pixel_overlap = collect_attribute([target_dimension], exptConfigDict) * overlap_proportion

    if float(pixel_overlap).is_integer():
        return int(pixel_overlap)
    else:
        # if not overlap is not a whole number in px
        closest_overlap = int(math.ceil(pixel_overlap))
        closest_overlap += closest_overlap % 2  # make even
        return closest_overlap


def collect_tiling_mode(exptConfigDict: Dict) -> str:

    tiling_mode = collect_attribute(["tilingMode"], exptConfigDict)

    if re.search("snake", tiling_mode, re.IGNORECASE):
        return "snake"
    elif re.search("grid", tiling_mode, re.IGNORECASE):
        return "grid"
    else:
        raise ValueError(f"Unknown tiling mode found: {tiling_mode}")


def create_cycle_channel_names(exptConfigDict: Dict) -> List[str]:
    num_channels = collect_attribute(["numChannels"], exptConfigDict)
    return [f"CH{i}" for i in range(1, num_channels + 1)]


def calc_how_many_big_concurrent_tasks(
    tile_size: Tuple[int, int],
    overlap_size: Tuple[int, int],
    dtype: np.dtype,
    n_tiles_per_plane: int,
) -> int:
    ram_stats = psutil.virtual_memory()
    num_cpus = psutil.cpu_count()
    free_ram_gb = ram_stats.available / 1024 ** 3
    img_dtype = int(re.search(r"(\d+)", np.dtype(dtype).name).groups()[0])  # int16 -> 16
    nbytes = img_dtype / 8

    img_plane_size = (
        n_tiles_per_plane
        * (tile_size[0] + overlap_size[0])
        * (tile_size[1] + overlap_size[1])
        * nbytes
    )
    img_plane_size += round(img_plane_size * 0.1)  # 10% overhead
    img_plane_size_gb = img_plane_size / 1024 ** 3

    num_of_concurrent_tasks = free_ram_gb // img_plane_size_gb
    if num_of_concurrent_tasks < 1:
        print(
            "WARNING: Image plane size is larger than memory. "
            + "Will try to run large jobs in a single process"
        )
        num_of_concurrent_tasks = 1
    if num_of_concurrent_tasks > num_cpus:
        num_of_concurrent_tasks = num_cpus
    return num_of_concurrent_tasks


def get_img_dtype(raw_data_location: Path) -> np.dtype:
    tile_dtype = get_tile_dtype(raw_data_location)
    return tile_dtype


def get_tile_shape_no_overlap(
    raw_data_location: Path,
    overlap_y: int,
    overlap_x: int,
) -> Tuple[int, int]:
    tile_shape_with_overlap = get_tile_shape(raw_data_location)
    tile_height = tile_shape_with_overlap[0] - overlap_y
    tile_width = tile_shape_with_overlap[1] - overlap_x
    return tile_height, tile_width


def standardize_metadata(directory: Path):
    experiment_json_files = find_files(
        directory,
        "experiment.json",
        ignore_processed_derived_metadata_dirs=True,
    )
    segmentation_json_files = find_files(
        directory,
        "segmentation.json",
        ignore_processed_derived_metadata_dirs=True,
    )
    segmentation_text_files = find_files(directory, "config.txt")
    channel_names_files = find_files(
        directory,
        "channelNames.txt",
        ignore_processed_derived_metadata_dirs=True,
    )
    channel_names_report_files = find_files(
        directory, "channelnames_report.csv", ignore_processed_derived_metadata_dirs=True
    )

    warn_if_multiple_files(segmentation_json_files, "segmentation JSON")
    warn_if_multiple_files(segmentation_text_files, "segmentation text")
    warn_if_multiple_files(channel_names_files, "channel names")
    warn_if_multiple_files(channel_names_report_files, "channel names report CSV")

    if not (segmentation_json_files or segmentation_text_files):
        raise ValueError("Segmentation parameters files not found. Cannot continue.")

    if segmentation_json_files and segmentation_text_files:
        message_pieces = ["Found segmentation JSON and text files. Using JSON.", "\tJSON:"]
        message_pieces.extend(f"\t\t{json_file}" for json_file in segmentation_json_files)
        message_pieces.append("\tText:")
        message_pieces.extend(f"\t\t{text_file}" for text_file in segmentation_text_files)

        logger.warning("\n".join(message_pieces))

    if len(experiment_json_files) > 1:
        message_pieces = ["Found multiple experiment JSON files:"]
        message_pieces.extend(f"\t{filename}" for filename in experiment_json_files)
        raise ValueError("\n".join(message_pieces))

    # Read in the experiment JSON config.
    experiment_json_file = experiment_json_files[0]
    with open(experiment_json_file, "r") as exptJsonFile:
        exptConfigDict = json.load(exptJsonFile)
    logger.info(f"Finished reading file {experiment_json_file}")

    # Read in the segmentation parameters. If we have a JSON file, use that.
    if segmentation_json_files:
        segmentation_json_file = segmentation_json_files[0]
        logger.info(f"Reading segmentation parameters from {segmentation_json_file}")
        with open(segmentation_json_file, "r") as segmJsonFile:
            segmParams = json.load(segmJsonFile)
    else:
        segmentation_text_file = min(
            segmentation_text_files,
            key=highest_file_sort_key,
        )
        logger.info(f"Reading segmentation parameters from {segmentation_text_file}")
        with open(segmentation_text_file, "r") as segmTextFile:
            segmParams = {}
            for line in segmTextFile:
                # Haven't seen any whitespace around '=', but be safe
                fieldName, fieldContents = (piece.strip() for piece in line.split("="))
                if fieldContents.isdigit():
                    fieldContents = int(fieldContents)
                segmParams[fieldName] = fieldContents

    logger.info("Finished reading segmentation parameters.")

    channel_names_qc_pass: Dict[str, List[str]] = defaultdict(list)
    if channel_names_report_files:
        channel_names_report_file = channel_names_report_files[0]
        with open(channel_names_report_file, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            for row in csvreader:
                channel_names_qc_pass[row[0]].append(row[1].lstrip())
    else:
        logger.warning(
            "No channelnames_report.csv file found. Including all channels in final output."
        )

    raw_data_location = find_raw_data_dir(directory)
    logger.info(f"Raw data location: {raw_data_location}")

    datasetInfo = {}

    datasetInfo["name"] = directory.name
    datasetInfo["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    datasetInfo["raw_data_location"] = fspath(raw_data_location.relative_to(directory))
    datasetInfo["channel_names_qc_pass"] = dict(channel_names_qc_pass)

    info_key_mapping = [
        ("emission_wavelengths", ["emission_wavelengths", "wavelengths"]),
        ("axial_resolution", ["zPitch", "z_pitch"]),
        ("lateral_resolution", ["xyResolution", "per_pixel_XY_resolution"]),
        ("magnification", ["magnification"]),
        ("num_z_planes", ["numZPlanes", "num_z_planes"]),
        ("numerical_aperture", ["aperture", "numerical_aperture"]),
        ("objective_type", ["objectiveType"]),
        ("region_height", ["region_height", "regionHeight"]),
        ("region_width", ["region_width", "regionWidth"]),
    ]

    for target_key, possibilities in info_key_mapping:
        datasetInfo[target_key] = collect_attribute(possibilities, exptConfigDict)

    try:
        datasetInfo["region_names"] = collect_attribute(["region_names", "regIdx"], exptConfigDict)
    except KeyError:
        # Not present in experiment configuration. Get from filesystem
        datasetInfo["region_names"] = get_region_names_from_directories(raw_data_location)

    # Get tile overlaps.
    tile_overlap_mappings = [
        ("tile_overlap_x", "tile_overlap_X"),
        ("tile_overlap_y", "tile_overlap_Y"),
    ]
    for target_key, possibleMatch in tile_overlap_mappings:
        try:
            datasetInfo[target_key] = collect_attribute([possibleMatch], exptConfigDict)
        except KeyError:
            datasetInfo[target_key] = calculate_pixel_overlaps_from_proportional(
                target_key, exptConfigDict
            )

    tile_shape = get_tile_shape_no_overlap(
        raw_data_location,
        datasetInfo["tile_overlap_y"],
        datasetInfo["tile_overlap_x"],
    )
    datasetInfo["tile_height"] = tile_shape[0]
    datasetInfo["tile_width"] = tile_shape[1]

    tile_dtype = get_tile_dtype(raw_data_location)
    datasetInfo["tile_dtype"] = tile_dtype.name

    # Get tiling mode.
    try:
        datasetInfo["tiling_mode"] = collect_attribute(["tiling_mode"], exptConfigDict)
    except KeyError:
        datasetInfo["tiling_mode"] = collect_tiling_mode(exptConfigDict)

    # Get per-cycle channel names.
    try:
        datasetInfo["per_cycle_channel_names"] = collect_attribute(
            ["channel_names"], exptConfigDict
        )
    except KeyError:
        datasetInfo["per_cycle_channel_names"] = create_cycle_channel_names(exptConfigDict)

    if channel_names_files:
        channel_names_file = min(
            channel_names_files,
            key=highest_file_sort_key,
        )
        logger.info(f"Reading channel names from {channel_names_file}")
        with open(channel_names_file, "r") as channelNamesFile:
            channelNames = channelNamesFile.read().splitlines()
            # HACK: work around odd 0x00 bytes read past where this file should end
            if ALL_ZERO_CHANNEL_NAME_PATTERN.match(channelNames[-1]):
                count = len(channelNames[-1])
                logger.info(f"Dropping spurious channel name containing {count} \\x00 characters")
                channelNames = channelNames[:-1]
    elif "channelNames" in exptConfigDict:
        logger.info("Obtaining channel names from configuration data")
        channelNames = collect_attribute(["channelNamesArray"], exptConfigDict["channelNames"])
    else:
        raise ValueError("Cannot find data for channel_names field.")

    # If there are identical channel names, make them unique by adding
    # incremental numbers to the end.
    channelNames = make_DAPI_channel_names_unique(channelNames)

    datasetInfo["channel_names"] = channelNames

    datasetInfo["num_cycles"] = int(
        len(channelNames) / len(datasetInfo["per_cycle_channel_names"])
    )

    bestFocusChannel = collect_attribute(
        ["bestFocusReferenceChannel", "best_focus_channel", "referenceChannel"], exptConfigDict
    )
    bestFocusCycle = collect_attribute(
        ["bestFocusReferenceCycle", "referenceCycle"], exptConfigDict
    )
    bestFocusChannelName = infer_channel_name_from_index(
        int(bestFocusCycle),
        int(bestFocusChannel),
        datasetInfo["channel_names"],
        len(datasetInfo["per_cycle_channel_names"]),
    )

    driftCompChannel = collect_attribute(
        ["driftCompReferenceChannel", "drift_comp_channel", "referenceChannel"], exptConfigDict
    )
    driftCompCycle = collect_attribute(
        ["driftCompReferenceCycle", "referenceCycle"], exptConfigDict
    )
    driftCompChannelName = infer_channel_name_from_index(
        int(driftCompCycle),
        int(driftCompChannel),
        datasetInfo["channel_names"],
        len(datasetInfo["per_cycle_channel_names"]),
    )

    datasetInfo["best_focus"] = bestFocusChannelName
    datasetInfo["drift_compensation"] = driftCompChannelName

    nucleiChannel = collect_attribute(["nuclearStainChannel"], segmParams)
    nucleiCycle = collect_attribute(["nuclearStainCycle"], segmParams)
    nucleiChannelName = infer_channel_name_from_index(
        int(nucleiCycle),
        int(nucleiChannel),
        datasetInfo["channel_names"],
        len(datasetInfo["per_cycle_channel_names"]),
    )

    # If we don't have a nuclei channel, we can't continue.
    if nucleiChannelName is None:
        raise ValueError("No nuclei stain channel found. Cannot continue.")

    membraneChannel = collect_attribute(["membraneStainChannel"], segmParams)
    membraneCycle = collect_attribute(["membraneStainCycle"], segmParams)
    membraneChannelName = infer_channel_name_from_index(
        int(membraneCycle),
        int(membraneChannel),
        datasetInfo["channel_names"],
        len(datasetInfo["per_cycle_channel_names"]),
    )

    datasetInfo["nuclei_channel"] = nucleiChannelName

    if membraneChannelName is not None:
        datasetInfo["membrane_channel"] = membraneChannelName

    # The target_shape needs to be worked out based on the metadata. See
    # comments on calculate_target_shape() function definition.
    datasetInfo["target_shape"] = calculate_target_shape(
        datasetInfo["magnification"],
        datasetInfo["tile_height"],
        datasetInfo["tile_width"],
    )

    overlap_size = (datasetInfo["tile_overlap_y"], datasetInfo["tile_overlap_x"])
    n_tiles_per_plane = datasetInfo["region_height"] * datasetInfo["region_width"]
    num_concurrent_tasks = calc_how_many_big_concurrent_tasks(
        tile_shape, overlap_size, tile_dtype, n_tiles_per_plane
    )
    datasetInfo["num_concurrent_tasks"] = num_concurrent_tasks
    pprint(datasetInfo)
    return datasetInfo


########
# MAIN #
########
if __name__ == "__main__":
    # Set up argument parser and parse the command line arguments.
    parser = argparse.ArgumentParser(
        description="Collect information required to perform analysis of a CODEX dataset, from various sources depending on submitted files. This script should be run manually after inspection of submission directories, and is hopefully only a temporary necessity until submission formats have been standardised."
    )
    parser.add_argument(
        "rawDataLocation",
        help="Path to directory containing raw data subdirectories (named with cycle and region numbers).",
        type=Path,
    )

    parser.add_argument(
        "convertedDataLocation",
        help="Path to directory containing with TIFF images converted from other file formats in subdirectories (named with cycle and region numbers).",
        type=Path,
    )

    parser.add_argument(
        "--outfile",
        type=Path,
    )

    args = parser.parse_args()

    converted_data_raw_dir = args.convertedDataLocation / "raw"
    is_converted_data_dir_empty = not any(converted_data_raw_dir.iterdir())
    if is_converted_data_dir_empty:
        data = standardize_metadata(args.rawDataLocation)
    else:
        data = standardize_metadata(args.convertedDataLocation)

    if not args.outfile:
        args.outfile = Path("pipelineConfig.json")

    ##############################
    # Write JSON pipeline config #
    ##############################
    logger.info("Writing pipeline config")
    with open(args.outfile, "w") as outfile:
        json.dump(data, outfile, indent=4)

    logger.info(f"Written pipeline config to {args.outfile}")
