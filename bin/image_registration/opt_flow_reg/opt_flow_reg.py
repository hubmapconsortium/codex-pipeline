import argparse
import gc
import os
import os.path as osp
import sys
from datetime import datetime
from typing import List, Tuple

import cv2 as cv
import dask
import numpy as np
import tifffile as tif

sys.path.append("/opt/image_registration")
from opt_flow_reg.metadata_handling import DatasetStructure
from opt_flow_reg.pyr_reg_of import PyrRegOF
from opt_flow_reg.warper import Warper

Image = np.ndarray


def read_and_max_project_pages(img_path: str, tiff_pages: List[int]):
    max_proj = tif.imread(img_path, key=tiff_pages[0])
    if len(tiff_pages) > 1:
        del tiff_pages[0]
        for p in tiff_pages:
            max_proj = np.maximum(max_proj, tif.imread(img_path, key=p))
    return cv.normalize(max_proj, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def warp_and_save_pages(TW, flow, in_path, meta, pages, warper):
    for p in pages:
        warper.image = tif.imread(in_path, key=p)
        warper.flow = flow
        warped_img = warper.warp()
        TW.write(
            warped_img,
            contiguous=True,
            photometric="minisblack",
            description=meta,
        )


def save_pages(TW, in_path, meta, pages):
    for p in pages:
        TW.write(
            tif.imread(in_path, key=p),
            contiguous=True,
            photometric="minisblack",
            description=meta,
        )


def register_and_save(
    in_path: str,
    out_dir: str,
    dataset_structure: dict,
    tile_size: int,
    overlap: int,
    num_pyr_lvl: int,
    num_iter: int,
    meta: str,
):
    """Read images and register them sequentially: 1<-2, 2<-3, 3<-4 etc.
    It is assumed that there is equal number of channels in each cycle.
    """
    registrator = PyrRegOF()
    registrator.tile_size = tile_size
    registrator.overlap = overlap
    registrator.num_pyr_lvl = num_pyr_lvl
    registrator.num_iterations = num_iter

    warper = Warper()
    warper.tile_size = tile_size
    warper.overlap = overlap

    filename = osp.basename(in_path).replace(".tif", "_opt_flow_registered.tif")
    out_path = osp.join(out_dir, filename)
    TW = tif.TiffWriter(out_path, bigtiff=True)

    for cyc in dataset_structure:
        this_cycle = dataset_structure[cyc]
        ref_ch_id = this_cycle["ref_channel_id"]
        print(f"Processing cycle {cyc + 1}/{len(dataset_structure)}")

        if cyc == 0:
            for ch in this_cycle["img_structure"]:
                pages = list(this_cycle["img_structure"][ch].values())
                save_pages(TW, in_path, meta, pages)

            ref_pages = list(this_cycle["img_structure"][ref_ch_id].values())
            ref_img = read_and_max_project_pages(in_path, ref_pages)
        else:
            mov_pages = list(this_cycle["img_structure"][ref_ch_id].values())
            mov_img = read_and_max_project_pages(in_path, mov_pages)

            registrator.ref_img = ref_img  # comes from previous cycle
            registrator.mov_img = mov_img
            flow = registrator.register()

            warper.image = mov_img
            warper.flow = flow
            ref_img = warper.warp()  # will be used in the next cycle

            print(f"Saving channels of cycle {cyc + 1}/{len(dataset_structure)}")
            for ch in this_cycle["img_structure"]:
                pages = list(this_cycle["img_structure"][ch].values())
                warp_and_save_pages(TW, flow, in_path, meta, pages, warper)
    TW.close()


def main(
    in_path: str,
    ref_channel: str,
    out_dir: str,
    n_workers: int = 1,
    tile_size: int = 1000,
    overlap: int = 100,
    num_pyr_lvl: int = 3,
    num_iter: int = 3,
):

    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    if n_workers == 1:
        dask.config.set({"scheduler": "synchronous"})
    else:
        dask.config.set({"num_workers": n_workers, "scheduler": "processes"})

    st = datetime.now()

    with tif.TiffFile(in_path, is_ome=True) as stack:
        ome_meta = stack.ome_metadata

    struc = DatasetStructure()
    struc.ref_channel_name = ref_channel
    struc.ome_meta_str = ome_meta
    dataset_structure = struc.get_dataset_structure()

    # perform registration of full stack
    register_and_save(
        in_path,
        out_dir,
        dataset_structure,
        tile_size,
        overlap,
        num_pyr_lvl,
        num_iter,
        ome_meta,
    )

    fin = datetime.now()
    print("time elapsed", fin - st)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="image stack to register")
    parser.add_argument("-c", type=str, required=True, help="channel for registration")
    parser.add_argument("-o", type=str, required=True, help="output dir")
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
        "--overlap",
        type=int,
        default=100,
        help="size of the overlap for one side of the image,"
        + "e.g. --overlap 50 = left,right,top,bottom overlaps are 50px each",
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
    args = parser.parse_args()

    main(
        args.i,
        args.c,
        args.o,
        args.n,
        args.tile_size,
        args.overlap,
        args.num_pyr_lvl,
        args.num_iter,
    )
