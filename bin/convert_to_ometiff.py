import argparse
import logging
import re
from multiprocessing import Pool
from os import walk
from pathlib import Path
from typing import List, Optional

import lxml.etree
import pandas as pd
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

structured_annotation_template="""<StructuredAnnotations>
<XMLAnnotation ID="Annotation:1">
    <Value>
        <OriginalMetadata>
            <Key>ProteinIDMap</Key>
            <Value>
                {protein_id_map_sa}
            </Value>
        </OriginalMetadata>
    </Value>
</XMLAnnotation>
</StructuredAnnotations>"""


def add_structured_annotations(xml_string: str, sa_str: str) -> str:
    sa_placement = xml_string.find("</Image>") + len("</Image>")
    xml_string = xml_string[:sa_placement] + sa_str + xml_string[sa_placement:]
    return xml_string


TIFF_FILE_NAMING_PATTERN = re.compile(r"^R\d{3}_X(\d{3})_Y(\d{3})\.tif")
metadata_filename_pattern = re.compile(r"^[0-9A-Fa-f]{32}antibodies\.tsv$")


def generate_sa_ch_info(ch_name: str, og_name: str, antb_info: Optional[pd.DataFrame], channel_id) -> str:
    empty_ch_info = (
        f'<Channel ID="{channel_id}" Name="{ch_name}" OriginalName="None" UniprotID="None" RRID="None" AntibodiesTsvID="None"/>'
    )
    if antb_info is None:
        ch_info = empty_ch_info
    else:
        if ch_name in antb_info["target"].to_list():
            ch_ind = antb_info[antb_info["target"] == ch_name].index[0]
            new_ch_name = antb_info.at[ch_ind, "target"]
            uniprot_id = antb_info.at[ch_ind, "uniprot_accession_number"]
            rr_id = antb_info.at[ch_ind, "rr_id"]
            antb_id = antb_info.at[ch_ind, "channel_id"]
            original_name = og_name
            ch_info = (
                f'<Channel ID="{channel_id}" Name="{new_ch_name}" OriginalName="{original_name}" UniprotID="{uniprot_id}" RRID="{rr_id}" AntibodiesTsvID="{antb_id}"/>'
            )
            return ch_info
        else:
            ch_info = empty_ch_info
    return ch_info


def generate_structured_annotations(ch_names: List[str], antb_info: Optional[pd.DataFrame], channel_id) -> str:
    ch_infos = []
    for ch_name in ch_names:
        ch_info = generate_sa_ch_info(ch_name, antb_info, channel_id)
        ch_infos.append(ch_info)
    ch_sa = "\n".join(ch_infos)
    sa = structured_annotation_template.format(protein_id_map_sa=ch_sa)
    return sa


def get_analyte_name(antibody_name: str) -> str:
    antb = re.sub(r"Anti-", "", antibody_name)
    antb = re.sub(r"\s+antibody", "", antb)
    return antb


def find_antibodies_meta(input_dir: Path) -> Optional[Path]:
    """
    Finds and returns the first metadata file for a HuBMAP data set.
    Does not check whether the dataset ID (32 hex characters) matches
    the directory name, nor whether there might be multiple metadata files.
    """
    # possible_dirs = [input_dir, input_dir / "extras"]
    metadata_filename_pattern = re.compile(r"^[0-9A-Za-z\-_]*antibodies\.tsv$")
    found_files = []
    for dirpath, dirnames, filenames in walk(input_dir):
        for filename in filenames:
            if metadata_filename_pattern.match(filename):
                found_files.append(Path(dirpath) / filename)

    if len(found_files) == 0:
        logger.warning("No antibody.tsv file found")
        antb_path = None
    else:
        antb_path = found_files[0]
    return antb_path


def sort_by_cycle(antb_path: Path):
    df = pd.read_table(antb_path)
    cycle_channel_pattern = re.compile(r"cycle(?P<cycle>\d+)_ch(?P<channel>\d+)", re.IGNORECASE)
    searches = [cycle_channel_pattern.search(v) for v in df["channel_id"]]
    cycles = [int(s.group("cycle")) for s in searches]
    channels = [int(s.group("channel")) for s in searches]
    df.index = [cycles, channels]
    df = df.sort_index()
    return df


def get_ch_info_from_antibodies_meta(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    # df = df.set_index("channel_id", inplace=False)
    antb_names = df["antibody_name"].to_list()
    antb_targets = [get_analyte_name(antb) for antb in antb_names]
    df["target"] = antb_targets
    return df


def map_cycles_and_channels(antibodies_df: pd.DataFrame) -> dict:
    channel_mapping = {
        channel_id.lower(): target
        for channel_id, target in zip(antibodies_df['channel_id'], antibodies_df['target'])
    }
    return channel_mapping


def replace_provider_ch_names_with_antb(og_ch_names_df: pd.DataFrame, antibodies_df: pd.DataFrame) -> List[str]:
    updated_channel_names = []
    mapping = map_cycles_and_channels(antibodies_df)
    for i in og_ch_names_df.index:
        channel_id = og_ch_names_df.at[i, 'channel_id'].lower()
        original_name = og_ch_names_df.at[i, 'channel_name']
        target = mapping.get(channel_id, None)
        if target is not None:
            updated_channel_names.append(target)
        else:
            updated_channel_names.append(original_name)
    return updated_channel_names


def get_original_names(og_ch_names_df: pd.DataFrame, antibodies_df: pd.DataFrame) -> List[str]:
    mapping = map_cycles_and_channels(antibodies_df)
    original_channel_names = []
    for i in og_ch_names_df.index:
        channel_id = og_ch_names_df.at[i, 'channel_id'].lower()
        original_name = og_ch_names_df.at[i, 'channel_name']
        target = mapping.get(channel_id, None)
        if target is not None:
            original_channel_names.append(original_name)
        else:
            original_channel_names.append("None")
    return original_channel_names


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

    # Separate channel and cycle info from channel names and remove "orig"
    cyc_ch_pattern = re.compile(r"cyc(\d+)_ch(\d+)_orig(.*)")
    og_ch_names_df = pd.DataFrame(channelList, columns=["Original_Channel_Name"])
    og_ch_names_df[['Cycle', 'Channel', 'channel_name']] = og_ch_names_df['Original_Channel_Name'].str.extract(cyc_ch_pattern)
    og_ch_names_df['channel_id'] = 'cycle' + og_ch_names_df['Cycle'] + '_ch' + og_ch_names_df['Channel']

    return og_ch_names_df


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

    channel_ids = [f"Channel:0:{i}" for i in range(len(channelNames))]
    original_channel_names = get_original_names(og_ch_names_df, antb_info)
    full_ch_info = ""
    for i in range(0, len(channelNames)):
        omeXml.image().Pixels.Channel(i).Name = channelNames[i]
        channel_id = channel_ids[i]
        omeXml.image().Pixels.Channel(i).ID = channel_id
        # Extract channel information for structured annotations
        ch_name = channelNames[i]
        original_name = original_channel_names[i]
        ch_info = generate_sa_ch_info(ch_name, original_name, antb_info, channel_id)
        full_ch_info = (full_ch_info + ch_info)
    struct_annot = add_structured_annotations(omeXml.to_xml('utf-8'), full_ch_info)
    with ome_tiff_writer.OmeTiffWriter(ometiffFile) as ome_writer:
        ome_writer.save(
            imageDataForOmeTiff,
            ome_xml=struct_annot,
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
                og_ch_names_df
            )
        )
    # Uncomment the next line to run as a series, comment the plural line
    # for argtuple in args_for_conversion :
    #    convert_tiff_file( argtuple )

    with Pool(processes=subprocesses) as pool:
        pool.imap_unordered(convert_tiff_file, args_for_conversion)
        pool.close()
        pool.join()


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
    antb_path = find_antibodies_meta(args.input_data_dir)

    lateral_resolution = get_lateral_resolution(args.cytokit_config)
    df = sort_by_cycle(antb_path)
    antb_info = get_ch_info_from_antibodies_meta(df)
    original_ch_names_df = collect_expressions_extract_channels(extractFileList[0])
    updated_channel_names = replace_provider_ch_names_with_antb(original_ch_names_df, antb_info)

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
        # For the extract, pull the correctly ordered list of channel names from
        # one of the files, as they aren't guaranteed to be in the same order as
        # the YAML config.
        # df = sort_by_cycle(antb_path)
        # antb_info = get_ch_info_from_antibodies_meta(df)
        # original_ch_names_df = collect_expressions_extract_channels(extractFileList[0])
        # updated_channel_names = replace_provider_ch_names_with_antb(original_ch_names_df, antb_info)
        create_ome_tiffs(
            extractFileList,
            output_dir / extract_expressions_piece / "ome-tiff",
            updated_channel_names,
            lateral_resolution,
            args.processes,
            antb_info,
            original_ch_names_df,
        )
