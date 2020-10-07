from pathlib import Path
import re
import argparse
import numpy as np
import tifffile as tif
import pandas as pd
from typing import List, Tuple, Union
import dask
Image = np.ndarray


def generate_ome_meta_for_mask(size_x: int, size_y: int, dtype):
        template = """<?xml version="1.0" encoding="utf-8"?>
            <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
              <Image ID="Image:0" Name="segmentation_mask_stitched.ome.tiff">
                
                <Pixels BigEndian="true" DimensionOrder="XYZCT" ID="Pixels:0" SizeC="4" SizeT="1" SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" Type="{dtype}">
                    <Channel ID="Channel:0:0" Name="cells" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:1" Name="nuclei" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:2" Name="cell_boundaries" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:3" Name="nucleus_boundaries" SamplesPerPixel="1" />

                    <TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="1" />
                    <TiffData FirstC="1" FirstT="0" FirstZ="0" IFD="1" PlaneCount="1" />
                    <TiffData FirstC="2" FirstT="0" FirstZ="0" IFD="2" PlaneCount="1" />
                    <TiffData FirstC="3" FirstT="0" FirstZ="0" IFD="3" PlaneCount="1" />
                </Pixels>

              </Image>
            </OME>
        """
        ome_meta = template.format(size_x=size_x, size_y=size_y, dtype=np.dtype(dtype).name)
        return ome_meta


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


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def path_to_dict(path: Path):
    """
    Extract region, x position, y position and put into the dictionary
    {R:region, X: position, Y: position, path: path}
    """
    value_list = re.split(r'(\d+)(?:_?)', path.name)[:-1]
    d = dict(zip(*[iter(value_list)]*2))
    d = {k: int(v) for k, v in d.items()}
    d.update({"path": path})
    return d


def get_slices(arr: np.ndarray, hor_f: int, hor_t: int, ver_f: int, ver_t: int, padding: dict, overlap=0):
    left_check  = hor_f - padding['left']
    top_check   = ver_f - padding['top']
    right_check = hor_t - arr.shape[-1]
    bot_check   = ver_t - arr.shape[-2]

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
        ver_t = arr.shape[0]

    big_image_slice = (slice(ver_f, ver_t), slice(hor_f, hor_t))
    block_shape = (ver_t - ver_f, hor_t - hor_f)
    block_slice = (slice(top_pad_size + overlap, block_shape[0] + overlap), slice(left_pad_size + overlap, block_shape[1] + overlap))

    return big_image_slice, block_slice


def stitch_plane(path_list: List[Path], page: int,
                 x_nblocks: int, y_nblocks: int,
                 block_shape: list, dtype,
                 overlap: int, padding: dict, remap_dict: dict = None) -> Tuple[Image, Union[np.ndarray, None]]:

    x_axis = -1
    y_axis = -2

    block_x_size = block_shape[x_axis] - overlap * 2
    block_y_size = block_shape[y_axis] - overlap * 2

    big_image_x_size = (x_nblocks * block_x_size) - padding["left"] - padding["right"]
    big_image_y_size = (y_nblocks * block_y_size) - padding["top"] - padding["bottom"]

    big_image_shape = (big_image_y_size, big_image_x_size)
    big_image = np.zeros(big_image_shape, dtype=dtype)

    previous_tile_max = 0
    tile_additions = np.zeros((y_nblocks, x_nblocks), dtype=dtype)
    print('n blocks x,y:', (x_nblocks, y_nblocks))
    print('plane shape x,y:', big_image_shape[::-1])
    n = 0
    for i in range(0, y_nblocks):
        ver_f = i * block_y_size
        ver_t = ver_f + block_y_size

        for j in range(0, x_nblocks):
            hor_f = j * block_x_size
            hor_t = hor_f + block_x_size

            big_image_slice, block_slice = get_slices(big_image, hor_f, hor_t, ver_f, ver_t, padding, overlap)
            block = tif.imread(path_to_str(path_list[n]), key=page).astype(dtype)

            if remap_dict is not None:
                block[np.nonzero(block)] += previous_tile_max

            big_image[tuple(big_image_slice)] = block[tuple(block_slice)]

            if remap_dict is not None:
                tile_additions[i, j] = previous_tile_max

                # update previous tile max
                non_zero_selection = block[np.nonzero(block)]
                if len(non_zero_selection) > 0:
                    previous_tile_max = non_zero_selection.max()

            n += 1
    if remap_dict is None:
        tile_additions = None
    return big_image, tile_additions


def _find_overlapping_border_labels(img1: Image, img2: Image, overlap: int, mode: str) -> dict:
    if mode == 'horizontal':
        img1_ov = img1[:, -overlap:]
        img2_ov = img2[:, :overlap]
    elif mode == 'vertical':
        img1_ov = img1[-overlap:, :]
        img2_ov = img2[:overlap, :]

    nrows, ncols = img2_ov.shape

    remap_dict = dict()

    for i in range(0, nrows):
        for j in range(0, ncols):
            old_value = img2_ov[i, j]
            if old_value in remap_dict:
                continue
            else:
                new_value = img1_ov[i, j]
                if old_value > 0 and new_value > 0:
                    remap_dict[old_value] = img1_ov[i, j]

    return remap_dict


def _get_map_of_overlapping_labels(path_list: List[Path], img1_id: int, img2_id: int, overlap: int, mode: str):
    # take only first channel
    img1 = tif.imread(path_to_str(path_list[img1_id]), key=0)
    img2 = tif.imread(path_to_str(path_list[img2_id]), key=0)
    remapping = _find_overlapping_border_labels(img1, img2, overlap, mode=mode)
    return (img2_id, remapping)


def get_remapping_of_border_labels(path_list: List[Path],
                                    x_nblocks: int, y_nblocks: int,
                                    overlap: int) -> dict:
    remap_dict = dict()
    htask = []
    # initialize remap_dict for all img ids
    for i in range(0, y_nblocks):
        for j in range(0, x_nblocks):
            img_id = i * x_nblocks + j
            remap_dict[img_id] = {'horizontal': {}, 'vertical': {}}

    for i in range(0, y_nblocks):
        for j in range(0, x_nblocks - 1):
            img1_id = i * x_nblocks + j
            img2h_id = i * x_nblocks + (j + 1)
            htask.append(dask.delayed(_get_map_of_overlapping_labels)(path_list, img1_id, img2h_id, overlap, 'horizontal'))

    hor_values = dask.compute(*htask, scheduler='processes')
    hor_values = list(hor_values)
    for remap in hor_values:
        remap_dict[remap[0]]['horizontal'] = remap[1]
    del hor_values

    vtask = []
    for i in range(0, y_nblocks - 1):
        for j in range(0, x_nblocks):
            img1_id = i * x_nblocks + j
            img2v_id = (i + 1) * x_nblocks + j
            vtask.append(dask.delayed(_get_map_of_overlapping_labels)(path_list, img1_id, img2v_id, overlap, 'vertical'))

    ver_values = dask.compute(*vtask, scheduler='processes')
    ver_values = list(ver_values)

    for remap in ver_values:
        remap_dict[remap[0]]['vertical'] = remap[1]

    return remap_dict


def remap_values(big_image: Image, remap_dict: dict,
                 tile_additions: np.ndarray, block_shape: list,
                 overlap: int, x_nblocks: int, y_nblocks: int) -> Image:
    print('remapping values')
    x_axis = -1
    y_axis = -2
    x_block_size = block_shape[x_axis] - overlap * 2
    y_block_size = block_shape[y_axis] - overlap * 2

    this_block_slice = [slice(None), slice(None)]

    n = 0
    for i in range(0, y_nblocks):
        yf = i * y_block_size
        yt = yf + y_block_size

        this_block_slice[y_axis] = slice(yf, yt)

        for j in range(0, x_nblocks):
            xf = j * x_block_size
            xt = xf + x_block_size

            this_block_slice[x_axis] = slice(xf, xt)

            this_block = big_image[tuple(this_block_slice)]
            try:
                hor_remap = remap_dict[n]['horizontal']
            except KeyError:
                hor_remap = {}
            try:
                ver_remap = remap_dict[n]['vertical']
            except KeyError:
                ver_remap = {}

            modified_x = False
            modified_y = False

            if hor_remap != {}:
                left_tile_addition = tile_additions[i, j - 1]
                this_tile_addition = tile_additions[i, j]
                for old_value, new_value in hor_remap.items():
                    this_block[this_block == old_value + this_tile_addition] = new_value + left_tile_addition
                modified_x = True

            if ver_remap != {}:
                top_tile_addition = tile_additions[i - 1, j]
                this_tile_addition = tile_additions[i, j]
                for old_value, new_value in ver_remap.items():
                    this_block[this_block == old_value + this_tile_addition] = new_value + top_tile_addition
                modified_y = True

            if modified_x or modified_y:
                big_image[tuple(this_block_slice)] = this_block
            else:
                this_tile_addition = tile_additions[i, j]
                this_block[np.nonzero(this_block)] += this_tile_addition
                big_image[tuple(this_block_slice)] = this_block

            n += 1
    return big_image


def main(img_dir: Path, out_path: Path, overlap: int, padding_str: str, is_mask: bool):

    padding_int = [int(i) for i in padding_str.split(',')]
    padding = {"left": padding_int[0], "right": padding_int[1], "top": padding_int[2], "bottom": padding_int[3]}

    img_paths = get_img_listing(img_dir)
    positions = [path_to_dict(p) for p in img_paths]
    df = pd.DataFrame(positions)
    df.sort_values(["R", "Y", "X"], inplace=True)

    x_nblocks = df["X"].max()
    y_nblocks = df["Y"].max()
    path_list = df["path"].to_list()

    with tif.TiffFile(path_to_str(path_list[0])) as TF:
        block_shape = list(TF.series[0].shape)
        npages = len(TF.pages)
        dtype = TF.series[0].dtype
        ome_meta = TF.ome_metadata

    big_image_x_size = (x_nblocks * (block_shape[-1] - overlap * 2)) - padding["left"] - padding["right"]
    big_image_y_size = (y_nblocks * (block_shape[-2] - overlap * 2)) - padding["top"] - padding["bottom"]

    if is_mask:
        print('getting values for remapping')
        remap_dict = get_remapping_of_border_labels(path_list, x_nblocks, y_nblocks, overlap)
        dtype = np.uint32
        ome_meta = generate_ome_meta_for_mask(big_image_x_size, big_image_y_size, dtype)
    else:
        remap_dict = None
        ome_meta = re.sub(r'\sSizeX="\d+"', ' SizeX="' + str(big_image_x_size) + '"', ome_meta)
        ome_meta = re.sub(r'\sSizeY="\d+"', ' SizeY="' + str(big_image_y_size) + '"', ome_meta)


    with tif.TiffWriter(path_to_str(out_path), bigtiff=True) as TW:
        for p in range(0, npages):
            print('\npage', p)
            print('stitching')
            plane, tile_additions = stitch_plane(path_list, p, x_nblocks, y_nblocks, block_shape, dtype, overlap, padding, remap_dict)
            if remap_dict is not None:
                plane = remap_values(plane, remap_dict, tile_additions, block_shape, overlap, x_nblocks, y_nblocks)
            TW.save(plane, photometric="minisblack", description=ome_meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=Path, required=True, help='path to directory with images')
    parser.add_argument('-o', type=Path, required=True, help='path to output file')
    parser.add_argument('-v', type=int, required=True, default=0, help='overlap size in pixels, default 0')
    parser.add_argument('-p', type=str, default='0,0,0,0',
                        help='image padding that should be removed, 4 comma separated numbers: left, right, top, bottom.' +
                             'Default: 0,0,0,0')
    parser.add_argument('--mask', action='store_true', help='use this flag if image is a binary mask')

    args = parser.parse_args()

    main(args.i, args.o, args.v, args.p, args.mask)
