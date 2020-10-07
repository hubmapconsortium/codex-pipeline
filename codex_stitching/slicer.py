import copy
import os
import os.path as osp
import argparse

import numpy as np
import tifffile as tif
import dask


def get_block(arr, hor_f: int, hor_t: int, ver_f: int, ver_t: int, overlap=0):
    hor_f -= overlap
    hor_t += overlap
    ver_f -= overlap
    ver_t += overlap

    left_check  = hor_f
    top_check   = ver_f
    right_check = hor_t - arr.shape[1]
    bot_check   = ver_t - arr.shape[0]

    left_pad_size = 0
    top_pad_size = 0
    right_pad_size = 0
    bot_pad_size = 0

    if left_check < 0:
        left_pad_size = abs(left_check)
        hor_f = 0
    if top_check < 0:
        top_pad_size = abs(top_check)
        ver_f = 0
    if right_check > 0:
        right_pad_size = right_check
        hor_t = arr.shape[1]
    if bot_check > 0:
        bot_pad_size = bot_check
        ver_t = arr.shape[0]

    block_slice = (slice(ver_f, ver_t), slice(hor_f, hor_t))
    block = arr[block_slice]
    padding = ((top_pad_size, bot_pad_size), (left_pad_size, right_pad_size))
    if max(padding) > (0, 0):
        block = np.pad(block, padding, mode='constant')
    return block


def split_by_size(arr: np.ndarray, region: int, zplane: int, channel: int, block_w: int, block_h: int, overlap: int):
    """ Splits image into blocks by size of block.
        block_w - block width
        block_h - block height
    """
    x_axis = -1
    y_axis = -2
    arr_width, arr_height = arr.shape[x_axis], arr.shape[y_axis]

    x_nblocks = arr_width // block_w if arr_width % block_w == 0 else (arr_width // block_w) + 1
    y_nblocks = arr_height // block_h if arr_height % block_h == 0 else (arr_height // block_h) + 1

    blocks = []
    img_names = []

    # row
    for i in range(0, y_nblocks):
        # height of this block
        ver_f = block_h * i
        ver_t = ver_f + block_h

        # col
        for j in range(0, x_nblocks):
            # width of this block
            hor_f = block_w * j
            hor_t = hor_f + block_w

            block = get_block(arr, hor_f, hor_t, ver_f, ver_t, overlap)

            blocks.append(block)
            name = '{region:d}_{tile:05d}_Z{zplane:03d}_CH{channel:d}.tif'.format(region=region,
                                                                                  tile=(i * x_nblocks) + (j + 1),
                                                                                  zplane=zplane,
                                                                                  channel=channel)
            img_names.append(name)

    return blocks, img_names


def slice_tiff(in_path: str, out_dir: str, block_size: int, overlap: int, region: int, channel: int, zplane: int):

    this_plane_split, this_plane_img_names = split_by_size(
        tif.imread(in_path), region=region, zplane=zplane, channel=channel, block_w=block_size, block_h=block_size, overlap=overlap
    )

    task = []
    for i, img in enumerate(this_plane_split):
        task.append(dask.delayed(tif.imwrite)(osp.join(out_dir, this_plane_img_names[i]), img, photometric='minisblack'))

    dask.compute(*task, scheduler='threads')

