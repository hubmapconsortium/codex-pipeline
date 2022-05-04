import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import tifffile as tif


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


# --------- METADATA PROCESSING -----------


def get_first_element_of_dict(dictionary: dict):
    dict_keys = list(dictionary.keys())
    first_key = dict_keys[0]
    return dictionary[first_key]


def digits_from_str(string: str):
    return [int(x) for x in re.split(r"(\d+)", string) if x.isdigit()]


def process_cycle_map(cycle_map: dict):
    cycle_names = list(cycle_map.keys())
    cycle_ids = [digits_from_str(name)[0] for name in cycle_names]
    new_cycle_map = dict()
    for i in range(0, len(cycle_ids)):
        this_cycle_name = cycle_names[i]
        this_cycle_id = cycle_ids[i]
        new_cycle_map[this_cycle_id] = cycle_map[this_cycle_name]

    # sort keys
    sorted_keys = sorted(new_cycle_map.keys())
    processed_cycle_map = dict()
    for k in sorted_keys:
        processed_cycle_map[k] = new_cycle_map[k]
    return processed_cycle_map


def get_image_dims(path: str):
    with tif.TiffFile(path) as TF:
        image_shape = list(TF.series[0].shape)
        image_dims = list(TF.series[0].axes)
    dims = ["Z", "Y", "X"]
    image_dimensions = dict()
    for d in dims:
        if d in image_dims:
            idx = image_dims.index(d)
            image_dimensions[d] = image_shape[idx]
        else:
            image_dimensions[d] = 1
    return image_dimensions


def get_dimensions_per_cycle(cycle_map: dict):
    dimensions_per_cycle = dict()
    for cycle in cycle_map:
        this_cycle_channels = cycle_map[cycle]
        this_cycle_channels_paths = list(this_cycle_channels.values())
        num_channels = len(this_cycle_channels_paths)
        first_channel_dims = get_image_dims(this_cycle_channels_paths[0])
        num_z_planes = (
            1 if first_channel_dims["Z"] == 1 else first_channel_dims["Z"] * num_channels
        )
        this_cycle_dims = {
            "SizeT": 1,
            "SizeZ": num_z_planes,
            "SizeC": num_channels,
            "SizeY": first_channel_dims["Y"],
            "SizeX": first_channel_dims["X"],
        }
        dimensions_per_cycle[cycle] = this_cycle_dims
    return dimensions_per_cycle


def generate_channel_meta(channel_names: List[str], cycle_id: int, offset: int):
    channel_elements = []
    for i, channel_name in enumerate(channel_names):
        channel_attrib = {
            "ID": "Channel:0:" + str(offset + i),
            "Name": channel_name,
            "SamplesPerPixel": "1",
        }
        channel = ET.Element("Channel", channel_attrib)
        channel_elements.append(channel)
    return channel_elements


def generate_tiffdata_meta(image_dimensions: dict):
    tiffdata_elements = []
    ifd = 0
    for t in range(0, image_dimensions["SizeT"]):
        for c in range(0, image_dimensions["SizeC"]):
            for z in range(0, image_dimensions["SizeZ"]):
                tiffdata_attrib = {
                    "FirstT": str(t),
                    "FirstC": str(c),
                    "FirstZ": str(z),
                    "IFD": str(ifd),
                }
                tiffdata = ET.Element("TiffData", tiffdata_attrib)
                tiffdata_elements.append(tiffdata)
                ifd += 1
    return tiffdata_elements


def image_dimensions_combined_for_all_cycles(image_dimensions_per_cycle: dict):
    combined_dimensions = dict()
    dimensions_that_change_in_cycles = ["SizeT", "SizeZ", "SizeC"]
    num_cycles = len(list(image_dimensions_per_cycle.keys()))
    first_cycle = get_first_element_of_dict(image_dimensions_per_cycle)
    for dim in dimensions_that_change_in_cycles:
        this_dim_value = first_cycle[dim]
        if this_dim_value == 1:
            combined_dim_value = 1
        else:
            combined_dim_value = this_dim_value * num_cycles

        combined_dimensions[dim] = combined_dim_value
    combined_dimensions["SizeY"] = first_cycle["SizeY"]
    combined_dimensions["SizeX"] = first_cycle["SizeX"]
    return combined_dimensions


def generate_default_pixel_attributes(image_path: str):
    with tif.TiffFile(image_path) as TF:
        img_dtype = TF.series[0].dtype

    pixels_attrib = {
        "ID": "Pixels:0",
        "DimensionOrder": "XYCZT",
        "Interleaved": "false",
        "Type": img_dtype.name,
    }
    return pixels_attrib


def generate_combined_ome_meta(cycle_map: dict, image_dimensions: dict, pixels_attrib: dict):
    proper_ome_attrib = {
        "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
    }

    channel_id_offset = 0
    channel_elements = []
    for i, cycle in enumerate(cycle_map):
        channel_names = cycle_map[cycle]
        num_channels = len(channel_names)
        this_cycle_channels = generate_channel_meta(channel_names, cycle, channel_id_offset)
        channel_elements.extend(this_cycle_channels)
        channel_id_offset += num_channels
    tiffdata_elements = generate_tiffdata_meta(image_dimensions)

    for key, val in image_dimensions.items():
        image_dimensions[key] = str(val)

    pixels_attrib.update(image_dimensions)

    node_ome = ET.Element("OME", proper_ome_attrib)
    node_image = ET.Element("Image", {"ID": "Image:0", "Name": "default.tif"})
    node_pixels = ET.Element("Pixels", pixels_attrib)

    for ch in channel_elements:
        node_pixels.append(ch)

    for td in tiffdata_elements:
        node_pixels.append(td)

    node_image.append(node_pixels)
    node_ome.append(node_image)

    xmlstr = ET.tostring(node_ome, encoding="utf-8", method="xml").decode("ascii")
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    ome_meta = xml_declaration + xmlstr
    return ome_meta


def generate_separated_ome_meta(
    cycle_map: dict, image_dimensions_per_cycle: dict, pixels_attrib: dict
):
    ome_meta_per_cycle = dict()
    for cycle in cycle_map:
        this_cycle_image_dimensions = copy.deepcopy(image_dimensions_per_cycle[cycle])
        this_cycle_cycle_map = {cycle: cycle_map[cycle]}
        this_cycle_ome_meta = generate_combined_ome_meta(
            this_cycle_cycle_map, this_cycle_image_dimensions, pixels_attrib
        )
        ome_meta_per_cycle[cycle] = this_cycle_ome_meta

    return ome_meta_per_cycle


# ------- IMAGE PROCESSING ------------


def save_cycles_combined_into_one_file(
    cycle_map: dict, out_dir: Path, file_name: str, ome_meta: str
):
    out_path = out_dir / file_name
    with tif.TiffWriter(out_path, bigtiff=True) as TW:
        for cycle, channels in cycle_map.items():
            for channel_path in channels.values():
                TW.save(tif.imread(channel_path), photometric="minisblack", description=ome_meta)
    return out_path


def save_cycle(input_path_list: List[str], out_path: Path, ome_meta: str):
    with tif.TiffWriter(path_to_str(out_path), bigtiff=True) as TW:
        for path in input_path_list:
            TW.save(tif.imread(path), photometric="minisblack", description=ome_meta)


def save_cycles_separated_per_file(
    cycle_map: dict, out_dir: Path, file_name_t: str, ome_meta_per_cycle: dict
):
    out_paths = []
    for cycle, channels in cycle_map.items():
        channel_paths = channels.values()
        ome_meta = ome_meta_per_cycle[cycle]
        file_name = file_name_t.format(cycle=cycle)
        out_path = out_dir / file_name
        save_cycle(channel_paths, out_path, ome_meta)
        out_paths.append(out_path)
    return out_paths


def get_first_channel_path(channels_per_cycle):
    first_cycle_channels = get_first_element_of_dict(channels_per_cycle)
    first_channel_path = list(first_cycle_channels.values())[0]
    return first_channel_path


def bundle_channel_images(
    channels_per_cycle: Dict[int, Dict[str, Path]], out_dir: Path, mode: str
) -> List[Path]:
    make_dir_if_not_exists(out_dir)

    print("Creating OME metadata")
    # sorted_channels_per_cycle = sort_dict(channels_per_cycle)

    first_channel_path = get_first_channel_path(channels_per_cycle)

    pixels_attrib = generate_default_pixel_attributes(first_channel_path)
    image_dimensions_per_cycle = get_dimensions_per_cycle(channels_per_cycle)

    print("Processing images")
    if mode == "combine":
        file_name = "cycles_combined.tif"
        image_dimensions = image_dimensions_combined_for_all_cycles(image_dimensions_per_cycle)
        ome_meta = generate_combined_ome_meta(channels_per_cycle, image_dimensions, pixels_attrib)
        out_path = save_cycles_combined_into_one_file(
            channels_per_cycle, out_dir, file_name, ome_meta
        )
        return [out_path]
    elif mode == "separate":
        file_name_t = "cycle{cycle:03d}.tif"
        ome_meta_per_cycle = generate_separated_ome_meta(
            channels_per_cycle, image_dimensions_per_cycle, pixels_attrib
        )
        out_paths = save_cycles_separated_per_file(
            channels_per_cycle, out_dir, file_name_t, ome_meta_per_cycle
        )
        return out_paths
    else:
        raise ValueError("Incorrect mode:" + str(mode) + ". Allowed options: separate, combine")
