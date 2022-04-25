import re
import xml.etree.ElementTree as ET
from io import StringIO
from typing import List

from tifffile import TiffFile

XML = ET.ElementTree


def str_to_xml(xmlstr: str):
    """Converts str to xml and strips namespaces"""
    it = ET.iterparse(StringIO(xmlstr))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition("}")
    root = it.root
    return root


def extract_channel_info(ome_xml: XML):
    channels = ome_xml.find("Image").find("Pixels").findall("Channel")
    channel_names = [ch.get("Name") for ch in channels]
    channel_ids = [ch.get("ID") for ch in channels]
    channel_fluors = []
    for ch in channels:
        if "Fluor" in ch.attrib:
            channel_fluors.append(ch.get("Fluor"))
    image_attribs = ome_xml.find("Image").find("Pixels").attrib
    nchannels = int(image_attribs.get("SizeC", 1))
    nzplanes = int(image_attribs.get("SizeZ", 1))
    return channels, channel_names, channel_ids, channel_fluors, nchannels, nzplanes


def extract_pixels_info(ome_xml: XML):
    dims = ["SizeX", "SizeY", "SizeC", "SizeZ", "SizeT"]
    sizes = ["PhysicalSizeX", "PhysicalSizeY"]
    pixels = ome_xml.find("Image").find("Pixels")
    pixels_info = dict()
    for d in dims:
        pixels_info[d] = int(pixels.get(d, 1))
    for s in sizes:
        pixels_info[s] = float(pixels.get(s, 1))
    return pixels_info


def strip_cycle_info(name):
    ch_name = re.sub(r"^(c|cyc|cycle)\d+(\s+|_)", "", name)  # strip start
    ch_name2 = re.sub(r"(-\d+)?(_\d+)?$", "", ch_name)  # strip end
    return ch_name2


def find_where_ref_channel(ome_xml: XML, ref_channel: str):
    """Find if reference channel is in fluorophores or channel names and return them"""
    channels, channel_names, channel_ids, channel_fluors, _, _ = extract_channel_info(
        ome_xml
    )

    ref_ch = strip_cycle_info(ref_channel)
    channel_fluors = [fluor.lower() for fluor in channel_fluors]
    channel_names = [name.lower() for name in channel_names]

    # strip cycle id from channel name and fluor name
    if channel_fluors != []:
        fluors = [strip_cycle_info(fluor) for fluor in channel_fluors]
    else:
        fluors = None
    names = [strip_cycle_info(name) for name in channel_names]

    # check if reference channel is present somewhere
    if ref_ch in names:
        matches = names
    elif fluors is not None and ref_ch in fluors:
        matches = fluors
    else:
        if fluors is not None:
            msg = (
                f"Incorrect reference channel {str(ref_ch)}. "
                + f"Available channel names: {str(set(names))}, fluors: {str(set(fluors))}"
            )
            raise ValueError(msg)
        else:
            msg = (
                f"Incorrect reference channel {str(ref_ch)}. "
                + f"Available channel names: {str(set(names))}"
            )
            raise ValueError(msg)
    return matches


def get_info_from_ome_meta(img_path: str, ref_channel: str, is_stack: bool):
    with TiffFile(img_path) as TF:
        ome_meta_str = TF.ome_metadata
    ome_xml = str_to_xml(ome_meta_str)
    matches = find_where_ref_channel(ome_xml, ref_channel)
    channels, _, _, _, nchannels, nzplanes = extract_channel_info(ome_xml)

    ref_channel_ids = [
        _id for _id, ch in enumerate(matches) if ch == strip_cycle_info(ref_channel)
    ]
    total_channels = len(channels)
    if is_stack:
        nchannels_per_cycle = ref_channel_ids[1] - ref_channel_ids[0]
        ncycles = total_channels // nchannels_per_cycle
    else:
        nchannels_per_cycle = total_channels
        ncycles = 1

    return ncycles, nchannels_per_cycle, nzplanes, ref_channel_ids[0]


def get_img_list_structure(img_paths: List[str], ref_channel: str):
    img_list_structure = dict()

    for cyc, path in enumerate(img_paths):
        _, nchannels, nzplanes, ref_channel_id = get_info_from_ome_meta(
            path, ref_channel, is_stack=False
        )

        img_structure = dict()
        img_list_structure[cyc] = dict()
        tiff_page = 0

        for ch in range(0, nchannels):
            img_structure[ch] = dict()
            for z in range(0, nzplanes):
                img_structure[ch][z] = tiff_page
                tiff_page += 1

        img_list_structure[cyc]["img_structure"] = img_structure
        img_list_structure[cyc]["ref_channel_id"] = ref_channel_id
        img_list_structure[cyc]["img_path"] = path
    return img_list_structure


def get_stack_structure(img_path: str, ref_channel: str):
    ncycles, nchannels, nzplanes, ref_channel_id = get_info_from_ome_meta(
        img_path, ref_channel, is_stack=True
    )

    stack_structure = dict()
    tiff_page = 0
    for cyc in range(0, ncycles):
        img_structure = dict()
        stack_structure[cyc] = dict()
        for ch in range(0, nchannels):
            img_structure[ch] = dict()
            for z in range(0, nzplanes):
                img_structure[ch][z] = tiff_page
                tiff_page += 1
        stack_structure[cyc]["img_structure"] = img_structure
        stack_structure[cyc]["ref_channel_id"] = ref_channel_id
        stack_structure[cyc]["img_path"] = img_path
    return stack_structure


def get_dataset_structure(img_paths: List[str], ref_channel: str, is_stack: bool):
    if is_stack:
        return get_stack_structure(img_paths[0], ref_channel)
    else:
        return get_img_list_structure(img_paths, ref_channel)


def generate_new_metadata(img_paths, target_shape):
    ncycles = len(img_paths)
    time = []
    planes = []
    channels = []
    metadata_list = []
    phys_size_x_list = []
    phys_size_y_list = []

    for i in range(0, len(img_paths)):
        with TiffFile(img_paths[i]) as TF:
            img_axes = list(TF.series[0].axes)
            img_shape = TF.series[0].shape
            ome_meta = TF.ome_metadata
            metadata_list.append(ome_meta)

    for meta in metadata_list:
        pixels_info = extract_pixels_info(str_to_xml(meta))
        time.append(pixels_info["SizeT"])
        planes.append(pixels_info["SizeZ"])
        channels.append(pixels_info["SizeC"])
        phys_size_x_list.append(pixels_info["PhysicalSizeX"])
        phys_size_y_list.append(pixels_info["PhysicalSizeY"])

    max_time = max(time)
    max_planes = max(planes)
    total_channels = sum(channels)
    max_phys_size_x = max(phys_size_x_list)
    max_phys_size_y = max(phys_size_y_list)

    sizes = {
        "SizeX": str(target_shape[1]),
        "SizeY": str(target_shape[0]),
        "SizeC": str(total_channels),
        "SizeZ": str(max_planes),
        "SizeT": str(max_time),
        "PhysicalSizeX": str(max_phys_size_x),
        "PhysicalSizeY": str(max_phys_size_y),
    }

    # use metadata from first image as reference metadata
    ref_xml = str_to_xml(metadata_list[0])

    # set proper ome attributes tags
    proper_ome_attribs = {
        "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
    }
    ref_xml.attrib.clear()

    for attr, val in proper_ome_attribs.items():
        ref_xml.set(attr, val)

    # set new dimension sizes
    for attr, size in sizes.items():
        ref_xml.find("Image").find("Pixels").set(attr, size)

    # remove old channels and tiffdata
    old_channels = ref_xml.find("Image").find("Pixels").findall("Channel")
    for ch in old_channels:
        ref_xml.find("Image").find("Pixels").remove(ch)

    tiffdata = ref_xml.find("Image").find("Pixels").findall("TiffData")
    if tiffdata is not None or tiffdata != []:
        for td in tiffdata:
            ref_xml.find("Image").find("Pixels").remove(td)

    # add new channels
    write_format = (
        "0" + str(len(str(ncycles)) + 1) + "d"
    )  # e.g. for number 5 format = 02d, result = 05
    channel_id = 0
    for i in range(0, ncycles):
        (
            channels,
            channel_names,
            channel_ids,
            channel_fluors,
            num_channels_per_cycle,
            num_zplanes_per_channel,
        ) = extract_channel_info(str_to_xml(metadata_list[i]))

        cycle_name = "c" + format(i + 1, write_format) + " "
        new_channel_names = [cycle_name + ch for ch in channel_names]

        for ch in range(0, len(channels)):
            new_channel_id = "Channel:0:" + str(channel_id)
            new_channel_name = new_channel_names[ch]
            channels[ch].set("Name", new_channel_name)
            channels[ch].set("ID", new_channel_id)
            ref_xml.find("Image").find("Pixels").append(channels[ch])
            channel_id += 1

    # add new tiffdata
    ifd = 0
    for t in range(0, max_time):
        for c in range(0, total_channels):
            for z in range(0, max_planes):
                ET.SubElement(
                    ref_xml.find("Image").find("Pixels"),
                    "TiffData",
                    dict(
                        FirstC=str(c),
                        FirstT=str(t),
                        FirstZ=str(z),
                        IFD=str(ifd),
                        PlaneCount=str(1),
                    ),
                )
                ifd += 1

    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    result_ome_meta = xml_declaration + ET.tostring(
        ref_xml, method="xml", encoding="utf-8"
    ).decode("ascii", errors="ignore")

    return result_ome_meta
