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
    tile_shape = (ver_t - ver_f, hor_t - hor_f)
    tile_slice = (slice(top_pad_size + overlap, tile_shape[0] + overlap),
                   slice(left_pad_size + overlap, tile_shape[1] + overlap))

    return big_image_slice, tile_slice


def stitch_plane(path_list: List[Path], page: int,
                 x_ntiles: int, y_ntiles: int,
                 tile_shape: list, dtype,
                 overlap: int, padding: dict, remap_dict: dict = None) -> Tuple[Image, Union[np.ndarray, None]]:

    x_axis = -1
    y_axis = -2

    tile_x_size = tile_shape[x_axis] - overlap * 2
    tile_y_size = tile_shape[y_axis] - overlap * 2

    big_image_x_size = (x_ntiles * tile_x_size) - padding["left"] - padding["right"]
    big_image_y_size = (y_ntiles * tile_y_size) - padding["top"] - padding["bottom"]

    big_image_shape = (big_image_y_size, big_image_x_size)
    big_image = np.zeros(big_image_shape, dtype=dtype)

    previous_tile_max = 0
    tile_additions = np.zeros((y_ntiles, x_ntiles), dtype=dtype)
    print('n tiles x,y:', (x_ntiles, y_ntiles))
    print('plane shape x,y:', big_image_shape[::-1])
    n = 0
    for i in range(0, y_ntiles):
        ver_f = i * tile_y_size
        ver_t = ver_f + tile_y_size

        for j in range(0, x_ntiles):
            hor_f = j * tile_x_size
            hor_t = hor_f + tile_x_size

            big_image_slice, tile_slice = get_slices(big_image, hor_f, hor_t, ver_f, ver_t, padding, overlap)
            tile = tif.imread(path_to_str(path_list[n]), key=page).astype(dtype)

            if remap_dict is not None:
                tile[np.nonzero(tile)] += previous_tile_max

            big_image[tuple(big_image_slice)] = tile[tuple(tile_slice)]

            if remap_dict is not None:
                tile_additions[i, j] = previous_tile_max

                # update previous tile max
                non_zero_selection = tile[np.nonzero(tile)]
                if len(non_zero_selection) > 0:
                    previous_tile_max = non_zero_selection.max()

            n += 1
    if remap_dict is None:
        tile_additions = None
    return big_image, tile_additions


def _find_overlapping_border_labels(img1: Image, img2: Image, overlap: int, mode: str) -> dict:
    if mode == 'horizontal':
        img1_ov = img1[:, -overlap * 2: -overlap]
        img2_ov = img2[:, :overlap]
    elif mode == 'vertical':
        img1_ov = img1[-overlap * 2: -overlap, :]
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
                                    x_ntiles: int, y_ntiles: int,
                                    overlap: int) -> dict:
    remap_dict = dict()
    htask = []
    # initialize remap_dict for all img ids
    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles):
            img_id = i * x_ntiles + j
            remap_dict[img_id] = {'horizontal': {}, 'vertical': {}}

    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles - 1):
            img1_id = i * x_ntiles + j
            img2h_id = i * x_ntiles + (j + 1)
            htask.append(dask.delayed(_get_map_of_overlapping_labels)(path_list, img1_id, img2h_id, overlap, 'horizontal'))

    hor_values = dask.compute(*htask, scheduler='processes')
    hor_values = list(hor_values)
    for remap in hor_values:
        remap_dict[remap[0]]['horizontal'] = remap[1]
    del hor_values

    vtask = []
    for i in range(0, y_ntiles - 1):
        for j in range(0, x_ntiles):
            img1_id = i * x_ntiles + j
            img2v_id = (i + 1) * x_ntiles + j
            vtask.append(dask.delayed(_get_map_of_overlapping_labels)(path_list, img1_id, img2v_id, overlap, 'vertical'))

    ver_values = dask.compute(*vtask, scheduler='processes')
    ver_values = list(ver_values)

    for remap in ver_values:
        remap_dict[remap[0]]['vertical'] = remap[1]

    return remap_dict


def remap_values(big_image: Image, remap_dict: dict,
                 tile_additions: np.ndarray, tile_shape: list,
                 overlap: int, x_ntiles: int, y_ntiles: int) -> Image:
    print('remapping values')
    x_axis = -1
    y_axis = -2
    x_tile_size = tile_shape[x_axis] - overlap * 2
    y_tile_size = tile_shape[y_axis] - overlap * 2

    this_tile_slice = [slice(None), slice(None)]

    n = 0
    for i in range(0, y_ntiles):
        yf = i * y_tile_size
        yt = yf + y_tile_size

        this_tile_slice[y_axis] = slice(yf, yt)

        for j in range(0, x_ntiles):
            xf = j * x_tile_size
            xt = xf + x_tile_size

            this_tile_slice[x_axis] = slice(xf, xt)

            this_tile = big_image[tuple(this_tile_slice)]
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
                    this_tile[this_tile == old_value + this_tile_addition] = new_value + left_tile_addition
                modified_x = True

            if ver_remap != {}:
                top_tile_addition = tile_additions[i - 1, j]
                this_tile_addition = tile_additions[i, j]
                for old_value, new_value in ver_remap.items():
                    this_tile[this_tile == old_value + this_tile_addition] = new_value + top_tile_addition
                modified_y = True

            if modified_x or modified_y:
                big_image[tuple(this_tile_slice)] = this_tile

            n += 1
    return big_image


def get_dataset_info(img_dir: Path):
    img_paths = get_img_listing(img_dir)
    positions = [path_to_dict(p) for p in img_paths]
    df = pd.DataFrame(positions)
    df.sort_values(['R', 'Y', 'X'], inplace=True)
    df.reset_index(inplace=True)

    region_ids = list(df['R'].unique())
    x_ntiles = df['X'].max()
    y_ntiles = df['Y'].max()

    path_list_per_region = []

    for r in region_ids:
        region_selection = df[df['R'] == r].index
        path_list = list(df.loc[region_selection, 'path'])
        path_list_per_region.append(path_list)

    return path_list_per_region, x_ntiles, y_ntiles


def main(img_dir: Path, out_path: Path, overlap: int, padding_str: str, is_mask: bool):

    padding_int = [int(i) for i in padding_str.split(',')]
    padding = {"left": padding_int[0], "right": padding_int[1], "top": padding_int[2], "bottom": padding_int[3]}

    path_list_per_region, x_ntiles, y_ntiles = get_dataset_info(img_dir)

    with tif.TiffFile(path_to_str(path_list_per_region[0][0])) as TF:
        tile_shape = list(TF.series[0].shape)
        npages = len(TF.pages)
        dtype = TF.series[0].dtype
        ome_meta = TF.ome_metadata

    big_image_x_size = (x_ntiles * (tile_shape[-1] - overlap * 2)) - padding["left"] - padding["right"]
    big_image_y_size = (y_ntiles * (tile_shape[-2] - overlap * 2)) - padding["top"] - padding["bottom"]

    if is_mask:
        print('\nGetting values for remapping')
        dtype = np.uint32
        ome_meta = generate_ome_meta_for_mask(big_image_x_size, big_image_y_size, dtype)

        border_map_per_region = []
        for path_list in path_list_per_region:
            remap_dict = get_remapping_of_border_labels(path_list, x_ntiles, y_ntiles, overlap)
            border_map_per_region.append(remap_dict)
    else:
        remap_dict = None
        ome_meta = re.sub(r'\sSizeX="\d+"', ' SizeX="' + str(big_image_x_size) + '"', ome_meta)
        ome_meta = re.sub(r'\sSizeY="\d+"', ' SizeY="' + str(big_image_y_size) + '"', ome_meta)

    reg_prefix = 'reg{r:d}_'
    for r, path_list in enumerate(path_list_per_region):
        new_path = out_path.parent.joinpath(reg_prefix.format(r=r+1) + out_path.name)
        with tif.TiffWriter(path_to_str(new_path), bigtiff=True) as TW:
            for p in range(0, npages):
                print('\npage', p)
                print('stitching')
                plane, tile_additions = stitch_plane(path_list, p, x_ntiles, y_ntiles, tile_shape, dtype, overlap,
                                                     padding, remap_dict)

                if is_mask:
                    border_map = border_map_per_region[r]
                    plane = remap_values(plane, border_map, tile_additions, tile_shape, overlap, x_ntiles, y_ntiles)
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
