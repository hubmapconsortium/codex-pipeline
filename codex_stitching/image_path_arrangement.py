from pathlib import Path
from typing import List, Dict, Set, Union, Tuple, Any
import re


def sort_each_level(dictionary: dict):
    sorted_dictionary = dict()

    sorted_cycle_keys = sorted(dictionary.keys())
    for cycle_key in sorted_cycle_keys:
        sorted_dictionary[cycle_key] = {}
        sorted_region_keys = sorted(dictionary[cycle_key].keys())
        for region_key in sorted_region_keys:
            sorted_dictionary[cycle_key][region_key] = {}
            sorted_channel_keys = sorted(dictionary[cycle_key][region_key].keys())
            for channel_key in sorted_channel_keys:
                sorted_dictionary[cycle_key][region_key][channel_key] = {}
                sorted_tile_keys = sorted(dictionary[cycle_key][region_key][channel_key].keys())
                for tile_key in sorted_tile_keys:
                    sorted_dictionary[cycle_key][region_key][channel_key][tile_key] = {}
                    sorted_plane_keys = sorted(dictionary[cycle_key][region_key][channel_key][tile_key].keys())
                    for plane_key in sorted_plane_keys:
                        plane_path = dictionary[cycle_key][region_key][channel_key][tile_key][plane_key]
                        sorted_dictionary[cycle_key][region_key][channel_key][tile_key][plane_key] = plane_path

    return sorted_dictionary



def alpha_num_order(string: str) -> str:
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return ''.join([format(int(x), '05d') if x.isdigit()
                    else x for x in re.split(r'(\d+)', string)])


def get_img_listing(in_dir: Path) -> List[Path]:
    allowed_extensions = ('.tif', '.tiff')
    listing = list(in_dir.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    img_listing = sorted(img_listing, key=lambda x: alpha_num_order(x.name))
    return img_listing


def extract_digits_from_string(string: str) -> List[int]:
    digits = [int(x) for x in re.split(r'(\d+)', string) if x.isdigit()]  # '1_00001_Z02_CH3' -> '1', '00001', '02', '3' -> [1,1,2,3]
    return digits


def arrange_listing_by_channel_tile_zplane(listing: List[Path]) -> Dict[int, Dict[int, Dict[int, Path]]]:
    tile_arrangement = dict()
    for file_path in listing:
        digits = extract_digits_from_string(file_path.name)
        tile = digits[1]
        zplane = digits[2]
        channel = digits[3]

        if channel in tile_arrangement:
            if tile in tile_arrangement[channel]:
                tile_arrangement[channel][tile].update({zplane: file_path})
            else:
                tile_arrangement[channel][tile] = {zplane: file_path}
        else:
            tile_arrangement[channel] = {tile: {zplane: file_path}}

    return tile_arrangement


def get_image_paths_arranged_in_dict(img_dir: Path) -> Dict[int, Dict[int, Dict[int, Path]]]:
    img_listing = get_img_listing(img_dir)
    arranged_listing = arrange_listing_by_channel_tile_zplane(img_listing)

    return arranged_listing


def extract_cycle_and_region_from_name(dir_name: str) -> Tuple[int, int]:
    region = 1
    if 'reg' in dir_name:
        match = re.search(r'reg(\d+)', dir_name, re.IGNORECASE)
        if match is not None:
            region = int(match.groups()[0])
    cycle = int(re.search(r'cyc(\d+)', dir_name, re.IGNORECASE).groups()[0])

    return cycle, region


def arrange_dirs_by_cycle_region(img_dirs: List[Path]) -> Dict[int, Dict[int, Path]]:
    cycle_region_dict = dict()
    for dir_path in img_dirs:
        dir_name = dir_path.name
        cycle, region = extract_cycle_and_region_from_name(str(dir_name))

        if cycle in cycle_region_dict:
            cycle_region_dict[cycle][region] = dir_path
        else:
            cycle_region_dict[cycle] = {region: dir_path}

    return cycle_region_dict


def create_listing_for_each_cycle_region(img_dirs: List[Path]) -> Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]]:
    """ Returns {cycle: {region: {channel: {tile: {zplane: path}}}}} """
    listing_per_cycle = dict()

    cycle_region_dict = arrange_dirs_by_cycle_region(img_dirs)
    for cycle, regions in cycle_region_dict.items():
        for region, dir_path in regions.items():
            arranged_listing = get_image_paths_arranged_in_dict(dir_path)
            if cycle in listing_per_cycle:
                listing_per_cycle[cycle][region] = arranged_listing
            else:
                listing_per_cycle[cycle] = {region: arranged_listing}
    sorted_listing = sort_each_level(listing_per_cycle)
    return sorted_listing

