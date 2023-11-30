import os.path as osp

import dask
import numpy as np
import tifffile as tif


def get_tile(arr, hor_f: int, hor_t: int, ver_f: int, ver_t: int, overlap=0):
    hor_f -= overlap
    hor_t += overlap
    ver_f -= overlap
    ver_t += overlap

    left_check = hor_f
    top_check = ver_f
    right_check = hor_t - arr.shape[1]
    bot_check = ver_t - arr.shape[0]

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

    tile_slice = (slice(ver_f, ver_t), slice(hor_f, hor_t))
    tile = arr[tile_slice]
    padding = ((top_pad_size, bot_pad_size), (left_pad_size, right_pad_size))
    if max(padding) > (0, 0):
        tile = np.pad(tile, padding, mode="constant")
    return tile


def split_by_size(
    arr: np.ndarray, region: int, zplane: int, channel: int, tile_w: int, tile_h: int, overlap: int
):
    """Splits image into tiles by size of tile.
    tile_w - tile width
    tile_h - tile height
    """
    x_axis = -1
    y_axis = -2
    arr_width, arr_height = arr.shape[x_axis], arr.shape[y_axis]

    x_ntiles = arr_width // tile_w if arr_width % tile_w == 0 else (arr_width // tile_w) + 1
    y_ntiles = arr_height // tile_h if arr_height % tile_h == 0 else (arr_height // tile_h) + 1

    tiles = []
    img_names = []

    # row
    for i in range(0, y_ntiles):
        # height of this tile
        ver_f = tile_h * i
        ver_t = ver_f + tile_h

        # col
        for j in range(0, x_ntiles):
            # width of this tile
            hor_f = tile_w * j
            hor_t = hor_f + tile_w

            tile = get_tile(arr, hor_f, hor_t, ver_f, ver_t, overlap)

            tiles.append(tile)
            name = "{region:d}_{tile:05d}_Z{zplane:03d}_CH{channel:d}.tif".format(
                region=region, tile=(i * x_ntiles) + (j + 1), zplane=zplane, channel=channel
            )
            img_names.append(name)

    return tiles, img_names


def slice_img(
    in_path: str,
    out_dir: str,
    tile_size: int,
    overlap: int,
    region: int,
    channel: int,
    zplane: int,
):
    this_plane_tiles, this_plane_img_names = split_by_size(
        tif.imread(in_path),
        region=region,
        zplane=zplane,
        channel=channel,
        tile_w=tile_size,
        tile_h=tile_size,
        overlap=overlap,
    )

    task = []
    for i, img in enumerate(this_plane_tiles):
        task.append(
            dask.delayed(tif.imwrite)(
                osp.join(out_dir, this_plane_img_names[i]), img, photometric="minisblack"
            )
        )

    dask.compute(*task, scheduler="threads")
