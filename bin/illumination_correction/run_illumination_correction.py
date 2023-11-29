import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

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
        try:
            this_image = tif.imread(str(path.absolute()))
        except Exception as excp:
            # do not raise from excp because the main process cannot instantiate excp
            raise RuntimeError(f"Error reading tiff image {path}: {excp}")
        imgs.append(this_image)
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


def read_darkfield_imgs(
    illum_cor_dir: Path, stack_paths: Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]
) -> Dict[int, Dict[int, Dict[int, Dict[int, ImgStack]]]]:
    per_zplane_darkfield = dict()
    stack_name_template = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}_Z{z:03d}.tif"
    for cycle in stack_paths:
        per_zplane_darkfield[cycle] = dict()
        for region in stack_paths[cycle]:
            per_zplane_darkfield[cycle][region] = dict()
            for channel in stack_paths[cycle][region]:
                per_zplane_darkfield[cycle][region][channel] = dict()
                for zplane, stack_path in stack_paths[cycle][region][channel].items():
                    stack_name = stack_name_template.format(
                        cyc=cycle, reg=region, ch=channel, z=zplane
                    )
                    darkfield_filename = "darkfield_" + stack_name
                    darkfield_path = illum_cor_dir / "darkfield" / darkfield_filename
                    darkfield = tif.imread(str(darkfield_path.absolute()))  # float32 0-1
                    per_zplane_darkfield[cycle][region][channel][zplane] = darkfield
    return per_zplane_darkfield


def apply_illum_cor(img: Image, flatfield: Image) -> Image:
    orig_dtype = img.dtype
    dtype_info = np.iinfo(orig_dtype)
    orig_minmax = (dtype_info.min, dtype_info.max)
    imgf = img.astype(np.float32)

    corrected_imgf = imgf / flatfield

    corrected_img = np.clip(np.round(corrected_imgf, 0), *orig_minmax).astype(orig_dtype)
    return corrected_img


def correct_and_save(img_path: Path, flatfield: Image, out_path: Path):
    corrected_img = apply_illum_cor(tif.imread(str(img_path.absolute())), flatfield)
    with tif.TiffWriter(str(out_path.absolute())) as TW:
        TW.write(corrected_img, photometric="minisblack")
    del corrected_img


def apply_flatfield_and_save(
    listing: Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]],
    flatfields: Dict[int, Dict[int, Dict[int, Dict[int, Image]]]],
    # darkfields: Dict[int, Dict[int, Dict[int, Dict[int, Image]]]],
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
                        # darkfield = darkfields[cycle][region][channel][zplane]
                        tasks.append(dask.delayed(correct_and_save)(path, flatfield, out_path))
    dask.compute(*tasks)


def organize_listing_by_cyc_reg_ch_zplane(
    listing: Dict[int, Dict[int, Dict[int, Dict[int, Dict[int, Path]]]]],
    tile_ids_to_use: Iterable[int],
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
                        if tile in tile_ids_to_use:
                            if zplane in new_arrangemnt[cycle][region][channel]:
                                new_arrangemnt[cycle][region][channel][zplane].append(path)
                            else:
                                new_arrangemnt[cycle][region][channel][zplane] = [path]
    return new_arrangemnt


def run_basic(basic_macro_path: Path, log_dir: Path):
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
    macro_filename = basic_macro_path.name
    run_log = (
        "Command:\n"
        + res.args
        + "\n\nSTDERR:\n"
        + res.stderr.decode("utf-8")
        + "\n\nSTDOUT:\n"
        + res.stdout.decode("utf-8")
    )
    log_filename = macro_filename + ".log"
    log_path = log_dir / log_filename
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(run_log)
    return


def run_all_macros(macro_paths: Dict[int, Dict[int, Dict[int, Dict[int, Path]]]], log_dir: Path):
    tasks = []
    for cycle in macro_paths:
        for region in macro_paths[cycle]:
            for channel in macro_paths[cycle][region]:
                for zplane, macro_path in macro_paths[cycle][region][channel].items():
                    tasks.append(dask.delayed(run_basic)(macro_path, log_dir))
    dask.compute(*tasks)


def check_illum_cor_images(
    illum_cor_dir: Path,
    log_dir: Path,
    zplane_listing: Dict[int, Dict[int, Dict[int, Dict[int, List[Path]]]]],
):
    cor_img_name_template = "{cor_type}_Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}_Z{z:03d}.tif"
    log_name_template = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}_Z{z:03d}.tif.ijm.log"
    imgs_present = []
    imgs_missing = []
    imgs_missing_logs = []
    for cycle in zplane_listing:
        for region in zplane_listing[cycle]:
            for channel in zplane_listing[cycle][region]:
                for zplane, macro_path in zplane_listing[cycle][region][channel].items():
                    flatfield_fn = cor_img_name_template.format(
                        cor_type="flatfield", cyc=cycle, reg=region, ch=channel, z=zplane
                    )
                    darkfield_fn = cor_img_name_template.format(
                        cor_type="darkfield", cyc=cycle, reg=region, ch=channel, z=zplane
                    )
                    flatfield_path = illum_cor_dir / "flatfield" / flatfield_fn
                    # darkfield_path = illum_cor_dir / "darkfield" / darkfield_fn
                    if flatfield_path.exists():
                        imgs_present.append((flatfield_fn))
                    else:
                        imgs_missing.append((flatfield_fn))
                        log_path = log_dir / log_name_template.format(
                            cyc=cycle, reg=region, ch=channel, z=zplane
                        )
                        with open(log_path, "r", encoding="utf-8") as f:
                            log_content = f.read()
                        imgs_missing_logs.append(log_content)
    if len(imgs_missing) > 0:
        msg = (
            "Probably there was an error while running BaSiC. "
            + "There is no image in one or more directories."
        )
        print(msg)

        for i in range(0, len(imgs_missing)):
            print("\nOne or both are missing:")
            print(imgs_missing[i])
            print("ImageJ log:")
            print(imgs_missing_logs[i])
        raise ValueError(msg)
    return


def select_which_tiles_to_use(
    n_tiles_y: int, n_tiles_x: int, tile_dtype: str, tile_size: Tuple[int, int]
) -> Set[int]:
    """Select every n-th tile, keeping the max size of the tile stack at 2GB"""
    n_tiles = n_tiles_y * n_tiles_x

    img_dtype = int(re.search(r"(\d+)", tile_dtype).groups()[0])  # int16 -> 16
    nbytes = img_dtype / 8

    # max 2GB
    single_tile_gb = tile_size[0] * tile_size[1] * nbytes / 1024**3
    max_num_tiles = round(2.0 // single_tile_gb)

    step = max(n_tiles // max_num_tiles, 1)
    if step < 2 and n_tiles > max_num_tiles:
        step = 2
    tile_ids = set(list(range(0, n_tiles, step)))
    return tile_ids


def main(data_dir: Path, pipeline_config_path: Path):
    img_stack_dir = Path("/output/image_stacks/")
    macro_dir = Path("/output/basic_macros")
    illum_cor_dir = Path("/output/illumination_correction/")
    corrected_img_dir = Path("/output/corrected_images")
    log_dir = Path("/output/logs")

    make_dir_if_not_exists(img_stack_dir)
    make_dir_if_not_exists(macro_dir)
    make_dir_if_not_exists(illum_cor_dir)
    make_dir_if_not_exists(corrected_img_dir)
    make_dir_if_not_exists(log_dir)

    dataset_info = load_dataset_info(pipeline_config_path)

    tile_dtype = dataset_info["tile_dtype"]

    num_workers = dataset_info["num_concurrent_tasks"]
    dask.config.set({"num_workers": num_workers, "scheduler": "processes"})

    raw_data_dir = dataset_info["dataset_dir"]
    img_dirs = get_input_img_dirs(Path(data_dir / raw_data_dir))
    print("Getting image listing")
    listing = create_listing_for_each_cycle_region(img_dirs)

    tile_size = (
        dataset_info["tile_height"] + dataset_info["overlap_y"],
        dataset_info["tile_width"] + dataset_info["overlap_x"],
    )
    n_tiles = dataset_info["num_tiles"]
    n_tiles_y = dataset_info["num_tiles_y"]
    n_tiles_x = dataset_info["num_tiles_x"]

    tile_ids_to_use = select_which_tiles_to_use(n_tiles_y, n_tiles_x, tile_dtype, tile_size)

    print(
        f"tile size: {str(tile_size)}",
        f"| number of tiles: {str(n_tiles)}",
        f"| using {str(len(tile_ids_to_use))} tiles to compute illumination correction",
    )
    zplane_listing = organize_listing_by_cyc_reg_ch_zplane(listing, tile_ids_to_use)

    print("Resaving images as stacks")
    stack_paths = resave_imgs_to_stacks(zplane_listing, img_stack_dir)
    print("Generating BaSiC macros")
    macro_paths = generate_basic_macro_for_each_stack(stack_paths, macro_dir, illum_cor_dir)
    print("Running estimation of illumination")
    run_all_macros(macro_paths, log_dir)
    check_illum_cor_images(illum_cor_dir, log_dir, zplane_listing)

    print("Applying illumination correction")
    flatfields = read_flatfield_imgs(illum_cor_dir, stack_paths)
    # darkfields = read_darkfield_imgs(illum_cor_dir, stack_paths)
    apply_flatfield_and_save(listing, flatfields, corrected_img_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with dataset directory")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )
    args = parser.parse_args()
    main(args.data_dir, args.pipeline_config_path)
