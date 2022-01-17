import re
import shutil
import sys

sys.path.append("/opt/")
from typing import List, Dict, Tuple
from pathlib import Path
from collect_dataset_info import find_raw_data_dir
from czi2tiff import convert_czi_to_tiles


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def alpha_num_order(string: str) -> str:
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def get_file_listing_by_extension(in_dir: Path, allowed_extensions: List[str]) -> List[Path]:
    listing = list(in_dir.iterdir())
    file_listing = [f for f in listing if f.suffix in tuple(allowed_extensions)]
    file_listing = sorted(file_listing, key=lambda x: alpha_num_order(x.name))
    return file_listing


def extract_cycle_and_region_from_name(
    dir_name: str, cycle_prefix: str, region_prefix: str
) -> Tuple[int, int]:
    matched_region = re.search(region_prefix, dir_name, re.IGNORECASE) is not None
    matched_cycle = re.search(cycle_prefix, dir_name, re.IGNORECASE) is not None
    if matched_region:
        region_pattern = region_prefix + r"(\d+)"
        region = int(re.search(region_pattern, dir_name, re.IGNORECASE).groups()[0])
    else:
        region = 1
    if matched_cycle:
        cycle_pattern = cycle_prefix + r"(\d+)"
        cycle = int(re.search(cycle_pattern, dir_name, re.IGNORECASE).groups()[0])
    else:
        raise ValueError("Could not infer cycle id from file name")
    return cycle, region


def arrange_files_by_cycle_region(
    file_list: List[Path], cycle_prefix: str, region_prefix: str
) -> Dict[int, Dict[int, Path]]:
    cycle_region_dict = dict()
    for file_path in file_list:
        dir_name = file_path.name
        cycle, region = extract_cycle_and_region_from_name(
            str(dir_name), cycle_prefix, region_prefix
        )
        if cycle in cycle_region_dict:
            cycle_region_dict[cycle][region] = file_path
        else:
            cycle_region_dict[cycle] = {region: file_path}
    if cycle_region_dict != {}:
        return cycle_region_dict
    else:
        raise ValueError("Could not find cycle and region ids from file names")


def convert_czi_files(raw_data_dir: Path, output_dir: Path):
    czi_listing = get_file_listing_by_extension(raw_data_dir, [".czi", ".CZI"])
    img_dir_template = "Cyc{cyc:03d}_reg{reg:03d}"
    cycle_prefix = "cyc"
    region_prefix = "reg"
    cycle_region_dict = arrange_files_by_cycle_region(czi_listing, cycle_prefix, region_prefix)
    for cyc in cycle_region_dict:
        for reg, czi_file_path in cycle_region_dict[cyc].items():
            img_dir = output_dir / img_dir_template.format(cyc=cyc, reg=reg)
            make_dir_if_not_exists(img_dir)
            convert_czi_to_tiles(czi_file_path, reg, img_dir)


def check_if_need_to_convert(raw_data_dir: Path):
    # only check for CZI files for now
    czi_listing = get_file_listing_by_extension(raw_data_dir, [".czi", ".CZI"])
    if len(czi_listing) > 0:
        return True
    else:
        return False


def copy_metadata(raw_data_dir: Path, output_dir: Path):
    file_list = get_file_listing_by_extension(raw_data_dir, [".json", ".txt", ".tsv", ".csv", ".yaml", ".yml"])
    for file_path in file_list:
        dst = output_dir / file_path.name
        shutil.copy(file_path, dst)


def main(data_dir: Path):
    raw_data_dir = find_raw_data_dir(data_dir)

    output_dir = Path("/output/converted_dataset/raw")
    make_dir_if_not_exists(output_dir)

    print("Checking if need to do an image file conversion")
    if check_if_need_to_convert(raw_data_dir) is True:
        print("Images are in CZI format, proceeding with conversion")
        convert_czi_files(raw_data_dir, output_dir)
        copy_metadata(raw_data_dir, output_dir)
    else:
        print("Do not need to convert, skipping step")
        return


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    start = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to the dataset directory with image files")
    args = parser.parse_args()

    main(args.data_dir)
    print("time elapsed", str(datetime.now() - start))
