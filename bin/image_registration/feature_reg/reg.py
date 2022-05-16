import argparse
import gc
import os
import os.path as osp
import re
import sys
from datetime import datetime
from typing import List

import cv2 as cv
import dask
import numpy as np
import pandas as pd
import tifffile as tif
from skimage.transform import AffineTransform, warp

sys.path.append("/opt/image_registration")
from feature_reg.metadata_handling import generate_new_metadata, get_dataset_structure
from feature_reg.pyr_reg import PyrReg

Image = np.ndarray


def alphaNumOrder(string):
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def save_param(img_paths, out_dir, transform_matrices_flat, padding, image_shape):
    transform_table = pd.DataFrame(transform_matrices_flat)
    for i in transform_table.index:
        dataset_name = "dataset_{id}_{name}".format(id=i + 1, name=os.path.basename(img_paths[i]))
        transform_table.loc[i, "name"] = dataset_name
    cols = transform_table.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    transform_table = transform_table[cols]
    for i in range(0, len(padding)):
        transform_table.loc[i, "left"] = padding[i][0]
        transform_table.loc[i, "right"] = padding[i][1]
        transform_table.loc[i, "top"] = padding[i][2]
        transform_table.loc[i, "bottom"] = padding[i][3]
        transform_table.loc[i, "width"] = image_shape[1]
        transform_table.loc[i, "height"] = image_shape[0]
    try:
        transform_table.to_csv(out_dir + "registration_parameters.csv", index=False)
    except PermissionError:
        transform_table.to_csv(out_dir + "registration_parameters_1.csv", index=False)


def calculate_padding_size(bigger_shape, smaller_shape):
    """Find difference between shapes of bigger and smaller image."""
    diff = bigger_shape - smaller_shape

    if diff == 1:
        dim1 = 1
        dim2 = 0
    elif diff % 2 != 0:
        dim1 = int(diff // 2)
        dim2 = int((diff // 2) + 1)
    else:
        dim1 = dim2 = int(diff / 2)

    return dim1, dim2


def pad_to_size(target_shape, img):
    if img.shape == target_shape:
        return img, (0, 0, 0, 0)
    else:
        left, right = calculate_padding_size(target_shape[1], img.shape[1])
        top, bottom = calculate_padding_size(target_shape[0], img.shape[0])
        return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0), (
            left,
            right,
            top,
            bottom,
        )


def read_and_max_project_pages(img_path: str, tiff_pages: List[int]):
    max_proj = tif.imread(img_path, key=tiff_pages[0])

    if len(tiff_pages) > 1:
        del tiff_pages[0]
        for p in tiff_pages:
            max_proj = np.maximum(max_proj, tif.imread(img_path, key=p))

    return cv.normalize(max_proj, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def estimate_registration_parameters(
    dataset_structure, ref_cycle_id, tile_size, num_pyr_lvl, num_iter
):
    padding = []
    transform_matrices = []
    img_shapes = []
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    img_paths = [dataset_structure[cyc]["img_path"] for cyc in dataset_structure]

    for i in range(0, len(img_paths)):
        with tif.TiffFile(img_paths[i]) as TF:
            img_shapes.append(TF.series[0].shape[-2:])

    max_size_x = max([s[1] for s in img_shapes])
    max_size_y = max([s[0] for s in img_shapes])
    target_shape = (max_size_y, max_size_x)

    ref_img_structure = dataset_structure[ref_cycle_id]["img_structure"]
    ref_img_ref_channel_id = dataset_structure[ref_cycle_id]["ref_channel_id"]
    ref_img_path = dataset_structure[ref_cycle_id]["img_path"]

    ref_img_tiff_pages = list(ref_img_structure[ref_img_ref_channel_id].values())
    ref_img = read_and_max_project_pages(ref_img_path, ref_img_tiff_pages)
    gc.collect()
    ref_img, pad = pad_to_size(target_shape, ref_img)
    gc.collect()
    padding.append(pad)
    # ref_features = get_features(ref_img, tile_size)
    gc.collect()

    ncycles = len(dataset_structure.keys())
    print("Registering images")
    registrator = PyrReg()
    registrator.ref_img = ref_img
    registrator.num_pyr_lvl = num_pyr_lvl
    registrator.num_iterations = num_iter
    registrator.tile_size = tile_size
    registrator.calc_ref_img_features()

    for cycle in dataset_structure:
        print("image {0}/{1}".format(cycle + 1, ncycles))
        img_structure = dataset_structure[cycle]["img_structure"]
        ref_channel_id = dataset_structure[cycle]["ref_channel_id"]
        img_path = dataset_structure[cycle]["img_path"]

        if cycle == ref_cycle_id:
            transform_matrices.append(identity_matrix)
        else:
            mov_img_tiff_pages = list(img_structure[ref_channel_id].values())
            mov_img = read_and_max_project_pages(img_path, mov_img_tiff_pages)
            gc.collect()
            mov_img, pad = pad_to_size(target_shape, mov_img)
            padding.append(pad)

            transform_matrix = registrator.register(mov_img)
            transform_matrices.append(transform_matrix)
            gc.collect()
    return transform_matrices, target_shape, padding


def transform_imgs(dataset_structure, out_dir, target_shape, transform_matrices, is_stack):
    print("Transforming images")
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    output_path = osp.join(out_dir, "out.tif")

    input_img_paths = [dataset_structure[cyc]["img_path"] for cyc in dataset_structure]

    if is_stack:
        with tif.TiffFile(input_img_paths[0]) as TF:
            old_meta = TF.ome_metadata
        new_meta = old_meta
    else:
        new_meta = generate_new_metadata(input_img_paths, target_shape)

    ncycles = len(dataset_structure.keys())
    nzplanes = {
        cyc: len(dataset_structure[cyc]["img_structure"][0].keys()) for cyc in dataset_structure
    }
    max_zplanes = max(nzplanes.values())

    TW = tif.TiffWriter(output_path, bigtiff=True)

    for cyc in dataset_structure:
        print("image {0}/{1}".format(cyc + 1, ncycles))
        img_path = dataset_structure[cyc]["img_path"]
        TF = tif.TiffFile(img_path)
        transform_matrix = transform_matrices[cyc]

        img_structure = dataset_structure[cyc]["img_structure"]
        for channel in img_structure:
            for zplane in img_structure[channel]:
                page = img_structure[channel][zplane]
                img = TF.asarray(key=page)
                original_dtype = img.dtype

                img, _ = pad_to_size(target_shape, img)
                gc.collect()
                if not np.array_equal(transform_matrix, identity_matrix):
                    homogenous_transform_matrix = np.append(transform_matrix, [[0, 0, 1]], axis=0)
                    inv_matrix = np.linalg.pinv(
                        homogenous_transform_matrix
                    )  # Using partial inverse to handle singular matrices
                    AT = AffineTransform(inv_matrix)
                    img = warp(img, AT, output_shape=img.shape, preserve_range=True).astype(
                        original_dtype
                    )
                    gc.collect()
                TW.write(img, contiguous=True, photometric="minisblack", description=new_meta)
                page += 1
                gc.collect()

                if nzplanes[cyc] < max_zplanes:
                    diff = max_zplanes - nzplanes[cyc]
                    empty_page = np.zeros_like(img)
                    for a in range(0, diff):
                        TW.write(
                            empty_page,
                            contiguous=True,
                            photometric="minisblack",
                            description=new_meta,
                        )
                    del empty_page
                    gc.collect()
                del img
                gc.collect()

        TF.close()
    TW.close()


def check_input_size(img_paths: List[str], is_stack: bool):
    if len(img_paths) == 1:
        if is_stack:
            pass
        else:
            raise ValueError("You need to provide at least two images to do a registration.")
    elif len(img_paths) > 1:
        if is_stack:
            raise ValueError(
                "Too many input images. " + "When flag --stack enabled only one image can be used"
            )
        else:
            pass
    else:
        raise ValueError("You need to provide at least two images to do a registration.")


def main(
    img_paths: list,
    ref_img_id: int,
    ref_channel: str,
    out_dir: str,
    n_workers: int = 1,
    tile_size: int = 1000,
    num_pyr_lvl: int = 3,
    num_iter: int = 3,
    stack: bool = False,
    estimate_only: bool = False,
    load_param: str = "none",
):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not out_dir.endswith("/"):
        out_dir = out_dir + "/"

    st = datetime.now()
    print("\nstarted", st)

    if n_workers == 1:
        dask.config.set({"scheduler": "synchronous"})
    else:
        dask.config.set({"num_workers": n_workers, "scheduler": "processes"})

    is_stack = stack
    ref_channel = ref_channel.lower()
    check_input_size(img_paths, is_stack)

    dataset_structure = get_dataset_structure(img_paths, ref_channel, is_stack)

    if load_param == "none":
        transform_matrices, target_shape, padding = estimate_registration_parameters(
            dataset_structure, ref_img_id, tile_size, num_pyr_lvl, num_iter
        )

    else:
        reg_param = pd.read_csv(load_param)
        target_shape = (reg_param.loc[0, "height"], reg_param.loc[0, "width"])

        transform_matrices = []
        padding = []
        for i in reg_param.index:
            matrix = (
                reg_param.loc[i, ["0", "1", "2", "3", "4", "5"]]
                .to_numpy()
                .reshape(2, 3)
                .astype(np.float32)
            )
            pad = reg_param.loc[i, ["left", "right", "top", "bottom"]].to_list()
            transform_matrices.append(matrix)
            padding.append(pad)

    if not estimate_only:
        transform_imgs(dataset_structure, out_dir, target_shape, transform_matrices, is_stack)

    transform_matrices_flat = [M.flatten() for M in transform_matrices]
    img_paths2 = [dataset_structure[cyc]["img_path"] for cyc in dataset_structure]
    save_param(img_paths2, out_dir, transform_matrices_flat, padding, target_shape)

    fin = datetime.now()
    print("\nelapsed time", fin - st)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image registration")

    parser.add_argument(
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="paths to images you want to register separated by space.",
    )
    parser.add_argument(
        "-r",
        type=int,
        required=True,
        help="reference image id, e.g. if -i 1.tif 2.tif 3.tif, and you ref image is 1.tif, then -r 0 (starting from 0)",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=True,
        help='reference channel name, e.g. DAPI. Enclose in double quotes if name consist of several words e.g. "Atto 490LS".',
    )
    parser.add_argument(
        "-o", type=str, required=True, help="directory to output registered image."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="multiprocessing: number of processes, default 1",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1000,
        help="size of a side of a square tile, "
        + "e.g. --tile_size 1000 = tile with dims 1000x1000px",
    )
    parser.add_argument(
        "--num_pyr_lvl",
        type=int,
        default=3,
        help="number of pyramid levels. Default 3, 0 - will not use pyramids",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=3,
        help="number of registration iterations per pyramid level. Default 3",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="add this flag if input is image stack instead of image list",
    )
    parser.add_argument(
        "--estimate_only",
        action="store_true",
        help="add this flag if you want to get only registration parameters and do not want to process images.",
    )
    parser.add_argument(
        "--load_param",
        type=str,
        default="none",
        help="specify path to csv file with registration parameters",
    )

    args = parser.parse_args()
    main(
        args.i,
        args.r,
        args.c,
        args.o,
        args.n,
        args.tile_size,
        args.num_pyr_lvl,
        args.num_iter,
        args.stack,
        args.estimate_only,
        args.load_param,
    )
