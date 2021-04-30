import re
from pathlib import Path
from typing import Dict, List, Tuple, Union


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}


def alpha_num_order(string: str) -> str:
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def get_img_listing(in_dir: Path) -> List[Path]:
    allowed_extensions = (".tif", ".tiff")
    listing = list(in_dir.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    img_listing = sorted(img_listing, key=lambda x: alpha_num_order(x.name))
    return img_listing


def extract_digits_from_string(string: str) -> List[int]:
    digits = [
        int(x) for x in re.split(r"(\d+)", string) if x.isdigit()
    ]  # '1_00001_Z02_CH3' -> '1', '00001', '02', '3' -> [1,1,2,3]
    return digits


def arrange_listing_by_channel_tile_zplane(
    listing: List[Path],
) -> Dict[int, Dict[int, Dict[int, Path]]]:
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


def extract_cycle_and_region_from_name(
    dir_name: str, cycle_prefix: str, region_prefix: str
) -> Tuple[Union[None, int], Union[None, int]]:
    matched_region = re.search(region_prefix, dir_name, re.IGNORECASE) is not None
    matched_cycle = re.search(cycle_prefix, dir_name, re.IGNORECASE) is not None
    if matched_region:
        region_pattern = region_prefix + r"(\d+)"
        region = int(re.search(region_pattern, dir_name, re.IGNORECASE).groups()[0])
    else:
        return None, None
    if matched_cycle:
        cycle_pattern = cycle_prefix + r"(\d+)"
        cycle = int(re.search(cycle_pattern, dir_name, re.IGNORECASE).groups()[0])
    else:
        return None, None
    return cycle, region


def arrange_dirs_by_cycle_region(
    img_dirs: List[Path], cycle_prefix: str, region_prefix: str
) -> Dict[int, Dict[int, Path]]:
    cycle_region_dict = dict()
    for dir_path in img_dirs:
        dir_name = dir_path.name
        cycle, region = extract_cycle_and_region_from_name(
            str(dir_name), cycle_prefix, region_prefix
        )
        if cycle is not None:
            if cycle in cycle_region_dict:
                cycle_region_dict[cycle][region] = dir_path
            else:
                cycle_region_dict[cycle] = {region: dir_path}
    if cycle_region_dict != {}:
        return cycle_region_dict
    else:
        raise ValueError("Could not find cycle and region directories")


def create_listing_for_each_cycle_region(
    img_dirs: List[Path],
) -> Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]]:
    """Returns {cycle: {region: {channel: {tile: {zplane: path}}}}}"""
    listing_per_cycle = dict()
    # Expected dir names Cyc1_reg1 or Cyc01_reg01
    cycle_prefix = "cyc"
    region_prefix = "reg"
    cycle_region_dict = arrange_dirs_by_cycle_region(img_dirs, cycle_prefix, region_prefix)
    for cycle, regions in cycle_region_dict.items():
        listing_per_cycle[cycle] = dict()
        for region, dir_path in regions.items():
            arranged_listing = get_image_paths_arranged_in_dict(dir_path)
            listing_per_cycle[cycle][region] = arranged_listing
    sorted_listing = sort_dict(listing_per_cycle)
    return sorted_listing
