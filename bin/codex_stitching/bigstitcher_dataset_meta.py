import xml.dom.minidom
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np


def convert_location(x, y):
    tile_loc = "1.0 0.0 0.0 {x} 0.0 1.0 0.0 {y} 0.0 0.0 1.0 0.0"
    return tile_loc.format(x=x, y=y)


def create_meta(file_pattern_str, num_tiles, tile_shape, tile_locations):
    root = ET.Element("SpimData", {"version": "0.2"})
    base_path = ET.SubElement(root, "BasePath", {"type": "relative"}).text = "."
    sequence_description = ET.SubElement(root, "SequenceDescription")

    # <ImageLoader>
    image_loader = ET.SubElement(
        sequence_description, "ImageLoader", {"format": "spimreconstruction.stack.loci"}
    )
    ET.SubElement(image_loader, "imagedirectory", {"type": "relative"}).text = "."
    ET.SubElement(image_loader, "filePattern").text = file_pattern_str
    ET.SubElement(image_loader, "layoutTimepoints").text = "0"
    ET.SubElement(image_loader, "layoutChannels").text = "0"
    ET.SubElement(image_loader, "layoutIlluminations").text = "0"
    ET.SubElement(image_loader, "layoutAngles").text = "0"
    ET.SubElement(image_loader, "layoutTiles").text = "1"
    ET.SubElement(image_loader, "imglib2container").text = "CellImgFactory"
    # </ImageLoader>
    # <ViewSetups>
    view_setups = ET.SubElement(sequence_description, "ViewSetups")

    view_setup_template = ET.Element("ViewSetup")
    ET.SubElement(view_setup_template, "id").text = "0"
    ET.SubElement(view_setup_template, "name").text = "0"
    ET.SubElement(view_setup_template, "size").text = "2048 2048 1"
    voxel_size = ET.SubElement(view_setup_template, "voxelSize")
    ET.SubElement(voxel_size, "unit").text = "um"
    ET.SubElement(voxel_size, "size").text = "1.0 1.0 1.0"
    view_attributes = ET.SubElement(view_setup_template, "attributes")
    ET.SubElement(view_attributes, "illumination").text = "0"
    ET.SubElement(view_attributes, "channel").text = "0"
    ET.SubElement(view_attributes, "tile").text = "0"
    ET.SubElement(view_attributes, "angle").text = "0"
    tile_shape_str = str(tile_shape[1]) + " " + str(tile_shape[0]) + " 1"
    for i in range(0, num_tiles):
        vs = deepcopy(view_setup_template)
        vs.find("id").text = str(i)
        vs.find("name").text = str(i)
        vs.find("size").text = tile_shape_str
        vs.find("attributes").find("tile").text = str(i)
        view_setups.append(vs)
    # </ViewSetups>
    # <Attributes>
    attrib_illumination = ET.SubElement(view_setups, "Attributes", {"name": "illumination"})
    attrib_illumination_illumination = ET.SubElement(attrib_illumination, "Illumination")
    ET.SubElement(attrib_illumination_illumination, "id").text = "0"
    ET.SubElement(attrib_illumination_illumination, "name").text = "0"

    attrib_channel = ET.SubElement(view_setups, "Attributes", {"name": "channel"})
    attrib_channel_channel = ET.SubElement(attrib_channel, "Channel")
    ET.SubElement(attrib_channel_channel, "id").text = "0"
    ET.SubElement(attrib_channel_channel, "name").text = "0"

    attrib_tile = ET.SubElement(view_setups, "Attributes", {"name": "tile"})

    attrib_tile_tile = ET.Element("Tile")
    ET.SubElement(attrib_tile_tile, "id").text = "0"
    ET.SubElement(attrib_tile_tile, "name").text = "0"
    ET.SubElement(attrib_tile_tile, "location").text = "0.0 0.0 0.0"
    for i in range(0, num_tiles):
        att = deepcopy(attrib_tile_tile)
        att.find("id").text = str(i)
        att.find("name").text = str(i + 1)
        attrib_tile.append(att)

    attrib_angle = ET.SubElement(view_setups, "Attributes", {"name": "angle"})
    attrib_angle_angle = ET.SubElement(attrib_angle, "Angle")
    ET.SubElement(attrib_angle_angle, "id").text = "0"
    ET.SubElement(attrib_angle_angle, "name").text = "0"
    # </Attributes>

    timepoints = ET.SubElement(sequence_description, "Timepoints", {"type": "pattern"})
    ET.SubElement(timepoints, "integerpattern")
    # </SequenceDescription>
    # <ViewRegistrations>
    view_registrations = ET.SubElement(root, "ViewRegistrations")

    view_registration_template = ET.Element("ViewRegistration", {"timepoint": "0", "setup": "0"})
    view_transform_translation = ET.SubElement(
        view_registration_template, "ViewTransform", {"type": "affine"}
    )
    ET.SubElement(view_transform_translation, "Name").text = "Translation to Regular Grid"
    ET.SubElement(
        view_transform_translation, "affine"
    ).text = "1.0 0.0 0.0 -2867.2 0.0 1.0 0.0 -1024.0 0.0 0.0 1.0 0.0"
    view_transform_calibration = ET.SubElement(
        view_registration_template, "ViewTransform", {"type": "affine"}
    )
    ET.SubElement(view_transform_calibration, "Name").text = "calibration"
    ET.SubElement(
        view_transform_calibration, "affine"
    ).text = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"

    for i in range(0, num_tiles):
        vr = deepcopy(view_registration_template)
        vr.set("timepoint", "0")
        vr.set("setup", str(i))
        vr.find("ViewTransform").find("affine").text = convert_location(*tile_locations[i])
        view_registrations.append(vr)

    # </ViewRegistrations>
    view_interest_points = ET.SubElement(root, "ViewInterestPoints")
    bounding_boxes = ET.SubElement(root, "BoundingBoxes")
    point_spread_functions = ET.SubElement(root, "PointSpreadFunctions")
    stitching_results = ET.SubElement(root, "StitchingResults")
    IntensityAdjustments = ET.SubElement(root, "IntensityAdjustments")

    declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    xml_str = ET.tostring(root, encoding="utf-8").decode()
    xml_str = declaration + xml_str

    return xml_str


def grid_to_snake(arr):
    nrows = arr.shape[0]
    new_arr = arr.copy()
    for i in range(0, nrows):
        if i % 2 != 0:
            new_arr[i, :] = new_arr[i, :][::-1]
    return new_arr


def generate_dataset_xml(
    x_ntiles: int,
    y_ntiles: int,
    tile_shape: Tuple[int, int],
    x_overlap: int,
    y_overlap: int,
    pattern_str: str,
    out_path: Path,
    is_snake=True,
):
    num_tiles = x_ntiles * y_ntiles

    loc_array = np.arange(0, y_ntiles * x_ntiles).reshape(y_ntiles, x_ntiles)
    img_sizes_x = np.zeros_like(loc_array)
    img_sizes_y = np.zeros_like(loc_array)

    for y in range(0, y_ntiles):
        y_size = tile_shape[0] - y_overlap
        for x in range(0, x_ntiles):
            x_size = tile_shape[1] - x_overlap

            img_sizes_x[y, x] = x_size
            img_sizes_y[y, x] = y_size

    img_positions_x = np.concatenate((np.zeros((y_ntiles, 1)), img_sizes_x[:, 1:]), axis=1)
    img_positions_y = np.concatenate((np.zeros((1, x_ntiles)), img_sizes_y[1:, :]), axis=0)

    img_positions_x = np.cumsum(img_positions_x, axis=1)
    img_positions_y = np.cumsum(img_positions_y, axis=0)

    if is_snake:
        img_positions_x = grid_to_snake(img_positions_x)
        img_positions_y = grid_to_snake(img_positions_y)

    tile_locations = list(zip(list(np.ravel(img_positions_x)), list(np.ravel(img_positions_y))))

    bs_xml = create_meta(pattern_str, num_tiles, tile_shape, tile_locations)

    dom = xml.dom.minidom.parseString(bs_xml)
    pretty_xml_as_string = dom.toprettyxml()

    with open(out_path, "w") as s:
        s.write(pretty_xml_as_string)
