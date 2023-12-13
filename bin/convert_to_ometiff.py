import argparse
import logging
import re
from multiprocessing import Pool
from os import walk
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from antibodies_tsv_util import antibodies_tsv_util as antb_tools
from ome_types.model import AnnotationRef, Map, MapAnnotation, StructuredAnnotationList
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


def generate_sa_ch_info(
    channel_id: str,
    og_ch_names_info: pd.Series,
    antb_info: Optional[pd.DataFrame],
) -> Optional[MapAnnotation]:
    if antb_info is None:
        return None
    cycle, channel = og_ch_names_info["Cycle"], og_ch_names_info["Channel"]
    try:
        antb_row = antb_info.loc[(cycle, channel), :]
    except KeyError:
        return None

    uniprot_id = antb_row["uniprot_accession_number"]
    rrid = antb_row["rr_id"]
    antb_id = antb_row["channel_id"]
    ch_key = Map.M(k="Channel ID", value=channel_id)
    name_key = Map.M(k="Name", value=antb_row["target"])
    og_name_key = Map.M(k="Original Name", value=og_ch_names_info["channel_name"])
    uniprot_key = Map.M(k="UniprotID", value=uniprot_id)
    rrid_key = Map.M(k="RRID", value=rrid)
    antb_id_key = Map.M(k="AntibodiesTsvID", value=antb_id)
    ch_info = Map([ch_key, name_key, og_name_key, uniprot_key, rrid_key, antb_id_key])
    annotation = MapAnnotation(value=ch_info)
    return annotation


def map_cycles_and_channels(antibodies_df: pd.DataFrame) -> dict:
    channel_mapping = {
        channel_id.lower(): target
        for channel_id, target in zip(antibodies_df["channel_id"], antibodies_df["target"])
    }
    return channel_mapping


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

    with TiffFile(str(extractFile.absolute())) as TF:
        ij_meta = TF.imagej_metadata
    numChannels = int(ij_meta["channels"])
    channelList = ij_meta["Labels"][0:numChannels]

    # Remove "proc_" from the start of the channel names.
    procPattern = re.compile(r"^proc_(.*)")
    channelList = [procPattern.match(channel).group(1) for channel in channelList]

    return channelList


def convert_tiff_file(funcArgs):
    """
    Given a tuple containing a source TIFF file path, a destination OME-TIFF path,
    a list of channel names, a float value for the lateral resolution in
    nanometres, convert the source TIFF file to OME-TIFF format, containing
    polygons for segmented cell shapes in the "ROI" OME-XML element.
    """
    sourceFile, ometiffFile, channelNames, lateral_resolution, antb_info, og_ch_names_df = funcArgs

    logger.info(f"Converting file: {str(sourceFile)}")

    image = AICSImage(sourceFile)
    imageDataForOmeTiff = image.get_image_data("TCZYX")
    imageName = f"Image: {sourceFile.name}"

    # Create OME-XML metadata using build_ome
    ome_writer = OmeTiffWriter()
    omeXml = ome_writer.build_ome(
        data_shapes=[(image.dims.T, image.dims.C, image.dims.Z, image.dims.Y, image.dims.X)],
        data_types=[image.dtype],
        dimension_order=["TCZYX"],
        channel_names=[channelNames],
        image_name=[imageName],
        physical_pixel_sizes=[image.physical_pixel_sizes],
    )
    annotations = StructuredAnnotationList()
    for i, (channel_obj, channel_name, og_ch_names_row) in enumerate(
        zip(
            omeXml.images[0].pixels.channels,
            channelNames,
            og_ch_names_df.iterrows(),
        )
    ):
        channel_id = f"Channel:0:{i}"
        channel_obj.name = channel_name
        channel_obj.id = channel_id
        ch_info = generate_sa_ch_info(channel_id, og_ch_names_row[1], antb_info)
        if ch_info is None:
            continue
        channel_obj.annotation_refs.append(AnnotationRef(id=ch_info.id))
        annotations.append(ch_info)

    omeXml.structured_annotations = annotations
    for i in omeXml.structured_annotations:
        print(i)
    ome_writer.save(
        data=imageDataForOmeTiff,
        uri=str(ometiffFile),
        ome_xml=omeXml,
        dimension_order="TCZYX",
        channel_names=channelNames,
    )

    logger.info(f"OME-TIFF file created: {ometiffFile}")


def create_ome_tiffs(
    file_list: List[Path],
    output_dir: Path,
    channel_names: List[str],
    lateral_resolution: float,
    subprocesses: int,
    antb_info: Optional[pd.DataFrame] = None,
    og_ch_names_df: Optional[pd.DataFrame] = None,
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
                antb_info,
                og_ch_names_df,
            )
        )
    # Uncomment the next line to run as a series, comment the plural line
    for argtuple in args_for_conversion:
        convert_tiff_file(argtuple)

    # with Pool(processes=subprocesses) as pool:
    #     pool.imap_unordered(convert_tiff_file, args_for_conversion)
    #     pool.close()
    #     pool.join()


def check_dir_is_empty(dir_path: Path):
    return not any(dir_path.iterdir())


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
        "cytokit_output",
        help="Path to output of `cytokit processor`",
        type=Path,
    )
    parser.add_argument(
        "bg_sub_tiles",
        help="Path to tiles with subtracted background",
        type=Path,
    )
    parser.add_argument(
        "cytokit_config",
        help="Path to Cytokit YAML config file",
        type=Path,
    )
    parser.add_argument(
        "input_data_dir",
        help="Path to the input dataset",
        type=Path,
    )
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of parallel OME-TIFF conversions to perform at once",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    print("Cytokit output:")
    print_directory_tree(args.cytokit_output)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    cytometry_tile_dir_piece = Path("cytometry/tile")
    extract_expressions_piece = Path("extract/expressions")
    processor_data_json_piece = Path("processor/data.json")

    cytometryTileDir = args.cytokit_output / cytometry_tile_dir_piece
    print("Cytometry tile directory:", cytometryTileDir)

    extractDir = args.cytokit_output / extract_expressions_piece
    print("Extract expressions directory:", extractDir)

    if not check_dir_is_empty(args.bg_sub_tiles):
        extractDir = args.bg_sub_tiles
        print(list(Path(args.bg_sub_tiles).iterdir()))
    else:
        extractDir = args.cytokit_output / extract_expressions_piece
    print("Extract expressions directory:", extractDir)

    segmentationFileList = collect_tiff_file_list(cytometryTileDir, TIFF_FILE_NAMING_PATTERN)
    extractFileList = collect_tiff_file_list(extractDir, TIFF_FILE_NAMING_PATTERN)
    antb_path = antb_tools.find_antibodies_meta(args.input_data_dir)

    lateral_resolution = get_lateral_resolution(args.cytokit_config)
    df = antb_tools.sort_by_cycle(antb_path)
    antb_info = antb_tools.get_ch_info_from_antibodies_meta(df)
    extractChannelNames = collect_expressions_extract_channels(extractFileList[0])
    original_ch_names_df = antb_tools.create_original_channel_names_df(extractChannelNames)
    updated_channel_names = antb_tools.replace_provider_ch_names_with_antb(
        original_ch_names_df, antb_info
    )

    # Create segmentation mask OME-TIFFs
    if segmentationFileList:
        create_ome_tiffs(
            segmentationFileList,
            output_dir / cytometry_tile_dir_piece / "ome-tiff",
            SEGMENTATION_CHANNEL_NAMES,
            lateral_resolution,
            args.processes,
            antb_info,
            original_ch_names_df,
        )

    # Create the extract OME-TIFFs.
    if extractFileList:
        create_ome_tiffs(
            extractFileList,
            output_dir / extract_expressions_piece / "ome-tiff",
            updated_channel_names,
            lateral_resolution,
            args.processes,
            antb_info,
            original_ch_names_df,
        )
