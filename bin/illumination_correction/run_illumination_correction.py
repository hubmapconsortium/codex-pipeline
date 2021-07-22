import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import cv2 as cv
import dask
import numpy as np
import tifffile as tif

sys.path.append("/opt/")
from generate_basic_macro import fill_in_basic_macro_template, save_macro

from pipeline_utils.dataset_listing import (
    create_listing_for_each_cycle_region,
    get_img_listing,
)
from pipeline_utils.pipeline_config_reader import load_dataset_info

ImgStack = np.ndarray  # 3d
Image = np.ndarray  # 2d


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def convert_np_cv_dtype(npdtype: np.dtype) -> int:
    np_cv_dtype_map = {
        np.dtype("float32"): cv.CV_32F,
        np.dtype("int32"): cv.CV_32S,
        np.dtype("uint16"): cv.CV_16U,
        np.dtype("uint8"): cv.CV_8U,
        np.dtype("int8"): cv.CV_8S,
        np.dtype("int16"): cv.CV_16S,
    }
    return np_cv_dtype_map[npdtype]


def get_input_img_dirs(data_dir: Path):
    img_dirs = list(data_dir.iterdir())
    return img_dirs


def read_imgs_to_stack(img_paths: List[Path]) -> ImgStack:
    imgs = []
    for path in img_paths:
        imgs.append(tif.imread(str(path.absolute())))
    img_stack = np.stack(imgs, axis=0)
    return img_stack


def save_stack(out_path: Path, stack: ImgStack):
    with tif.TiffWriter(out_path) as TW:
        TW.save(stack, contiguous=True, photometric="minisblack")


def read_and_save_to_stack(path_list: List[Path], out_stack_path: Path):
    save_stack(out_stack_path, read_imgs_to_stack(path_list))


def resave_imgs_to_stacks(
    zplane_img_listing: Dict[int, Dict[int, Dict[int, Dict[int, List[Path]]]]], img_stack_dir: Path
) -> Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]:
    stack_paths = dict()
    stack_name_template = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}_Z{z:03d}.tif"
    tasks = []
    for cycle in zplane_img_listing:
        stack_paths[cycle] = dict()
        for region in zplane_img_listing[cycle]:
            stack_paths[cycle][region] = dict()
            for channel in zplane_img_listing[cycle][region]:
                stack_paths[cycle][region][channel] = dict()
                for zplane, path_list in zplane_img_listing[cycle][region][channel].items():
                    stack_name = stack_name_template.format(
                        cyc=cycle, reg=region, ch=channel, z=zplane
                    )
                    out_stack_path = img_stack_dir / stack_name
                    stack_paths[cycle][region][channel][zplane] = out_stack_path
                    tasks.append(dask.delayed(read_and_save_to_stack)(path_list, out_stack_path))
    dask.compute(*tasks)
    return stack_paths


def generate_basic_macro_for_each_stack(
    stack_paths: Dict[int, Dict[int, Dict[int, Dict[int, Path]]]],
    macro_out_dir: Path,
    illum_cor_dir: Path,
) -> Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]:
    macro_paths = dict()
    for cycle in stack_paths:
        macro_paths[cycle] = dict()
        for region in stack_paths[cycle]:
            macro_paths[cycle][region] = dict()
            for channel in stack_paths[cycle][region]:
                macro_paths[cycle][region][channel] = dict()
                for zplane, stack_path in stack_paths[cycle][region][channel].items():
                    macro_path = macro_out_dir / (stack_path.name + ".ijm")
                    macro = fill_in_basic_macro_template(stack_path, illum_cor_dir)
                    save_macro(macro_path, macro)
                    macro_paths[cycle][region][channel][zplane] = macro_path
    return macro_paths


def run_basic(basic_macro_path: Path):
    # It is expected that ImageJ is added to system PATH
    if platform.system() == "Windows":
        imagej_name = "ImageJ-win64"
    elif platform.system() == "Linux":
        imagej_name = "ImageJ-linux64"
    elif platform.system() == "Darwin":
        imagej_name = "ImageJ-macosx"

    command = imagej_name + " --headless --console -macro " + str(basic_macro_path)
    print("Started running BaSiC for", str(basic_macro_path))
    res = subprocess.run(
        command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if res.returncode == 0:
        print("Finished", str(basic_macro_path))
    else:
        raise Exception(
            "There was an error while running the BaSiC for "
            + str(basic_macro_path)
            + "\n"
            + res.stderr.decode("utf-8")
        )


def run_all_macros(macro_paths: Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]):
    tasks = []
    for cycle in macro_paths:
        for region in macro_paths[cycle]:
            for channel in macro_paths[cycle][region]:
                for zplane, macro_path in macro_paths[cycle][region][channel].items():
                    tasks.append(dask.delayed(run_basic)(macro_path))
    dask.compute(*tasks)


def read_flatfield_imgs(
    illum_cor_dir: Path, stack_paths: Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]
) -> Dict[int, Dict[int, Dict[int, Dict[int, ImgStack]]]]:
    per_zplane_flatfield = dict()
    stack_name_template = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}_Z{z:03d}.tif"
    for cycle in stack_paths:
        per_zplane_flatfield[cycle] = dict()
        for region in stack_paths[cycle]:
            per_zplane_flatfield[cycle][region] = dict()
            for channel in stack_paths[cycle][region]:
                per_zplane_flatfield[cycle][region][channel] = dict()
                for zplane, stack_path in stack_paths[cycle][region][channel].items():
                    stack_name = stack_name_template.format(
                        cyc=cycle, reg=region, ch=channel, z=zplane
                    )
                    flatfield_filename = "flatfield_" + stack_name
                    flatfield_path = illum_cor_dir / "flatfield" / flatfield_filename
                    flatfield = tif.imread(str(flatfield_path.absolute()))  # float32 0-1
                    per_zplane_flatfield[cycle][region][channel][zplane] = flatfield
    return per_zplane_flatfield


def apply_flatfield_cor(img: Image, flatfield: Image) -> Image:
    orig_dtype = img.dtype
    orig_range = (img.min(), img.max())
    cv_dtype = convert_np_cv_dtype(orig_dtype)
    imgf = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    corrected_imgf = imgf / flatfield

    new_range = (
        round(orig_range[0] * corrected_imgf.min()),
        round(orig_range[1] * corrected_imgf.max()),
    )
    corrected_img = cv.normalize(
        corrected_imgf, None, new_range[0], new_range[1], cv.NORM_MINMAX, cv_dtype
    )
    return corrected_img


def correct_and_save(img_path: Path, flatfield: Image, out_path: Path):
    corrected_img = apply_flatfield_cor(tif.imread(str(img_path.absolute())), flatfield)
    tif.imwrite(out_path, corrected_img)


def apply_flatfield_and_save(
    listing: Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]],
    flatfields: Dict[int, Dict[int, Dict[int, Dict[int, Image]]]],
    out_dir: Path,
):
    img_dir_template = "Cyc{cyc:03d}_reg{reg:03d}"
    img_name_template = "{reg:d}_{tile:05d}_Z{z:03d}_CH{ch:d}.tif"
    tasks = []
    for cycle in listing:
        for region in listing[cycle]:
            for channel in listing[cycle][region]:
                for tile, zplane_dict in listing[cycle][region][channel].items():
                    for zplane, path in zplane_dict.items():
                        img_dir_name = img_dir_template.format(cyc=cycle, reg=region)
                        img_name = img_name_template.format(
                            reg=region, tile=tile, z=zplane, ch=channel
                        )
                        out_dir_full = Path(out_dir / img_dir_name)
                        make_dir_if_not_exists(out_dir_full)
                        out_path = out_dir_full / img_name
                        flatfield = flatfields[cycle][region][channel][zplane]
                        tasks.append(dask.delayed(correct_and_save)(path, flatfield, out_path))
    dask.compute(*tasks)


def organize_listing_by_cyc_reg_ch_zplane(
    listing: Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]]
) -> Dict[int, Dict[int, Dict[int, Dict[int, List[Path]]]]]:
    new_arrangemnt = dict()
    for cycle in listing:
        new_arrangemnt[cycle] = dict()
        for region in listing[cycle]:
            new_arrangemnt[cycle][region] = dict()
            for channel in listing[cycle][region]:
                new_arrangemnt[cycle][region][channel] = dict()
                for tile, zplane_dict in listing[cycle][region][channel].items():
                    for zplane, path in zplane_dict.items():
                        if zplane in new_arrangemnt[cycle][region][channel]:
                            new_arrangemnt[cycle][region][channel][zplane].append(path)
                        else:
                            new_arrangemnt[cycle][region][channel][zplane] = [path]
    return new_arrangemnt


def main(data_dir: Path, pipeline_config_path: Path):
    """It is expected that images are separated
    into different directories per region, cycle, channel
    e.g. Cyc1_Reg1_Ch1/0001.tif
    """
    img_stack_dir = Path("/output/image_stacks/")
    macro_dir = Path("/output/basic_macros")
    illum_cor_dir = Path("/output/illumination_correction/")
    corrected_img_dir = Path("/output/corrected_images")

    make_dir_if_not_exists(img_stack_dir)
    make_dir_if_not_exists(macro_dir)
    make_dir_if_not_exists(illum_cor_dir)
    make_dir_if_not_exists(corrected_img_dir)

    dask.config.set({"num_workers": 10, "scheduler": "processes"})

    dataset_info = load_dataset_info(pipeline_config_path)
    raw_data_dir = dataset_info["dataset_dir"]
    img_dirs = get_input_img_dirs(Path(data_dir / raw_data_dir))
    listing = create_listing_for_each_cycle_region(img_dirs)

    zplane_listing = organize_listing_by_cyc_reg_ch_zplane(listing)

    stack_paths = resave_imgs_to_stacks(zplane_listing, img_stack_dir)

    macro_paths = generate_basic_macro_for_each_stack(stack_paths, macro_dir, illum_cor_dir)
    run_all_macros(macro_paths)

    flatfields = read_flatfield_imgs(illum_cor_dir, stack_paths)
    apply_flatfield_and_save(listing, flatfields, corrected_img_dir)

    print(list(corrected_img_dir.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with dataset directory")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )
    args = parser.parse_args()
    main(args.data_dir, args.pipeline_config_path)
