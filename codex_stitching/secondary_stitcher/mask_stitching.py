from typing import Dict, List, Tuple

import dask
import numpy as np

Image = np.ndarray


def generate_ome_meta_for_mask(size_y: int, size_x: int, dtype) -> str:
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
    ome_meta = template.format(size_y=size_y, size_x=size_x, dtype=np.dtype(dtype).name)
    return ome_meta


def remove_labels(
    img: Image, y_slice: slice, x_slice: slice, exclude_start: bool
) -> Tuple[Image, List[int]]:
    exclude_from_val_to_remove = [0]

    val_to_remove = []
    if y_slice != slice(None):
        img_slice_y = (y_slice, slice(None))
        val_to_remove_y = np.unique(img[img_slice_y]).tolist()
        val_to_remove.extend(val_to_remove_y)

    if x_slice != slice(None):
        img_slice_x = (slice(None), x_slice)
        val_to_remove_x = np.unique(img[img_slice_x]).tolist()
        val_to_remove.extend(val_to_remove_x)

    val_to_remove = set(sorted(val_to_remove))

    if exclude_start:
        if y_slice.start is None and x_slice.start is None:
            raise ValueError("Exclude start is enabled but slice start is None")
        exclusions = []
        if y_slice.start is not None:
            line_slice_y = (slice(y_slice.start, y_slice.start + 1), x_slice)
            exclusions.extend(np.unique(img[line_slice_y]).tolist())
        if x_slice.start is not None:
            line_slice_x = (y_slice, slice(x_slice.start, x_slice.start + 1))
            exclusions.extend(np.unique(img[line_slice_x]).tolist())

        unique_exclusions = sorted(set(exclusions))
        exclude_from_val_to_remove.extend(unique_exclusions)

    exclude_from_val_to_remove = set(sorted(exclude_from_val_to_remove))
    val_to_remove = [val for val in val_to_remove if val not in exclude_from_val_to_remove]

    img_copy = img.copy()
    for val in val_to_remove:
        img_copy[img_copy == val] = 0
    return img_copy, val_to_remove


def remove_overlapping_labels(img: Image, overlap: int, mode: str) -> Tuple[Image, List[int]]:
    left = (slice(None), slice(None, overlap))
    right = (slice(None), slice(-overlap, None))
    top = (slice(None, overlap), slice(None))
    bottom = (slice(-overlap, None), slice(None))

    mod_img = img.copy()
    excluded_labels = []
    if "left" in mode:
        mod_img, ex_lab = remove_labels(mod_img, *left, exclude_start=False)
        excluded_labels.extend(ex_lab)
    if "right" in mode:
        mod_img, ex_lab = remove_labels(mod_img, *right, exclude_start=True)
        excluded_labels.extend(ex_lab)
    if "top" in mode:
        mod_img, ex_lab = remove_labels(mod_img, *top, exclude_start=False)
        excluded_labels.extend(ex_lab)
    if "bottom" in mode:
        mod_img, ex_lab = remove_labels(mod_img, *bottom, exclude_start=True)
        excluded_labels.extend(ex_lab)
    excluded_labels = sorted(set(excluded_labels))
    return mod_img, excluded_labels


def reset_label_ids(img: Image) -> Image:
    unique_vals, indices = np.unique(img, return_inverse=True)

    unique_val_list = unique_vals.tolist()

    new_vals = list(range(0, len(unique_val_list)))

    new_unique_vals = np.array(new_vals)
    reset_img = new_unique_vals[indices].reshape(img.shape)
    return reset_img


def find_and_remove_overlapping_labels_in_first_channel(
    tiles: List[Image], y_ntiles: int, x_ntiles: int, overlap: int
) -> Tuple[List[Image], Dict[int, Dict[int, int]]]:
    excluded_labels = dict()
    modified_tiles = []
    task = []
    n = 0
    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles):

            label_remove_mode = ""
            if i == 0:
                label_remove_mode += " bottom "
            elif i == y_ntiles - 1:
                label_remove_mode += " top "
            else:
                label_remove_mode += " top bottom "
            if j == 0:
                label_remove_mode += " right "
            elif j == x_ntiles - 1:
                label_remove_mode += " left "
            else:
                label_remove_mode += " left right "

            task.append(
                dask.delayed(remove_overlapping_labels)(tiles[n], overlap, label_remove_mode)
            )
            n += 1
    computed_modifications = dask.compute(*task)
    for i, mod in enumerate(computed_modifications):
        modified_tiles.append(mod[0])
        excluded_labels[i] = {lab: 0 for lab in mod[1]}

    return modified_tiles, excluded_labels


def remove_overlapping_labels_in_another_channel(
    tiles: List[Image], excluded_labels: dict
) -> List[Image]:
    def exclude_labels(tile, labels):
        for lab in labels:
            tile[tile == lab] = 0
        return tile

    task = []
    for i in range(0, len(tiles)):
        task.append(dask.delayed(exclude_labels)(tiles[i], excluded_labels[i]))
    modified_tiles = dask.compute(*task)
    return list(modified_tiles)


def find_overlapping_border_labels(
    img1: Image, img2: Image, overlap: int, mode: str
) -> Dict[int, int]:
    if mode == "horizontal":
        img1_ov = img1[:, -overlap:]
        img2_ov = img2[:, overlap : overlap * 2]
    elif mode == "vertical":
        img1_ov = img1[-overlap:, :]
        img2_ov = img2[overlap : overlap * 2, :]
    else:  # horizontal+vertical
        img1_ov = img1[-overlap:, -overlap:]
        img2_ov = img2[overlap : overlap * 2, overlap : overlap * 2]

    nrows, ncols = img2_ov.shape

    border_map = dict()

    for i in range(0, nrows):
        for j in range(0, ncols):
            old_value = img2_ov[i, j]
            if old_value in border_map:
                continue
            else:
                new_value = img1_ov[i, j]
                if old_value > 0 and new_value > 0:
                    border_map[old_value] = new_value

    return border_map


def replace_overlapping_border_labels(
    img1: Image, img2: Image, overlap: int, mode: str
) -> Tuple[Image, Dict[int, int]]:
    border_map = find_overlapping_border_labels(img1, img2, overlap, mode)
    for old_value, new_value in border_map.items():
        img2[img2 == old_value] = new_value
    return img2, border_map


def find_and_replace_overlapping_border_labels_in_first_channel(
    tiles: List[Image], y_ntiles: int, x_ntiles: int, overlap: int, dtype
) -> Tuple[List[Image], Dict[int, Dict[int, int]], List[int]]:
    previous_tile_max = 0
    tile_ids = np.arange(0, y_ntiles * x_ntiles).reshape((y_ntiles, x_ntiles))
    modified_tiles = []
    tile_additions = []
    border_maps = dict()
    n = 0
    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles):

            tile = tiles[n]
            tile = tile.astype(dtype)
            this_tile_max = tile.max()
            tile_additions.append(previous_tile_max)
            tile[np.nonzero(tile)] += previous_tile_max

            if i != 0:
                top_tile_id = tile_ids[i - 1, j]
            else:
                top_tile_id = None
            if j != 0:
                left_tile_id = tile_ids[i, j - 1]
            else:
                left_tile_id = None
            if i != 0 and j != 0:
                top_left_tile_id = tile_ids[i - 1, j - 1]
            else:
                top_left_tile_id = None

            this_tile_border_map = dict()
            if top_tile_id is not None:
                tile, border_map = replace_overlapping_border_labels(
                    modified_tiles[top_tile_id], tile, overlap, "vertical"
                )
                this_tile_border_map.update(border_map)
            if left_tile_id is not None:
                tile, border_map = replace_overlapping_border_labels(
                    modified_tiles[left_tile_id], tile, overlap, "horizontal"
                )
                this_tile_border_map.update(border_map)
            if top_left_tile_id is not None:
                tile, border_map = replace_overlapping_border_labels(
                    modified_tiles[top_left_tile_id], tile, overlap, "horizontal+vertical"
                )
                this_tile_border_map.update(border_map)

            modified_tiles.append(tile)
            border_maps[n] = this_tile_border_map
            previous_tile_max += this_tile_max
            n += 1
    return modified_tiles, border_maps, tile_additions


def replace_overlapping_border_labels_in_another_channel(
    tiles: List[Image], border_maps: Dict[int, dict], tile_additions: List[int], dtype
) -> List[Image]:
    def replace_values(tile, value_map, addition, dtype):
        modified_tile = tile.astype(dtype)
        modified_tile[np.nonzero(modified_tile)] += addition
        if value_map != {}:
            for old_value, new_value in value_map.items():
                modified_tile[modified_tile == old_value] = new_value
            return modified_tile
        else:
            return modified_tile

    task = []
    for i, tile in enumerate(tiles):
        task.append(dask.delayed(replace_values)(tile, border_maps[i], tile_additions[i], dtype))
    modified_tiles = dask.compute(*task)

    return list(modified_tiles)


def update_old_values(
    excluded_labels: dict, tile_additions: List[int]
) -> Dict[int, Dict[int, int]]:
    upd_excluded_labels = dict()
    for tile in excluded_labels:
        this_tile_excluded_labels = dict()
        for old_value, new_value in excluded_labels[tile].items():
            upd_old_value = old_value + tile_additions[tile]
            this_tile_excluded_labels[upd_old_value] = new_value
        upd_excluded_labels[tile] = this_tile_excluded_labels
    return upd_excluded_labels


def modify_tiles_first_channel(
    tiles: List[Image], y_ntiles: int, x_ntiles: int, overlap: int, dtype
) -> Tuple[List[Image], Dict[int, Dict[int, int]], Dict[int, Dict[int, int]], List[int]]:
    mod_tiles, excluded_labels = find_and_remove_overlapping_labels_in_first_channel(
        tiles, y_ntiles, x_ntiles, overlap
    )
    (
        mod_tiles,
        border_maps,
        tile_additions,
    ) = find_and_replace_overlapping_border_labels_in_first_channel(
        mod_tiles, y_ntiles, x_ntiles, overlap, dtype
    )

    return mod_tiles, excluded_labels, border_maps, tile_additions


def modify_tiles_another_channel(
    tiles: List[Image], excluded_labels: dict, border_maps: dict, tile_additions: list, dtype
) -> List[Image]:
    mod_tiles = remove_overlapping_labels_in_another_channel(tiles, excluded_labels)
    if border_maps != {}:
        mod_tiles = replace_overlapping_border_labels_in_another_channel(
            mod_tiles, border_maps, tile_additions, dtype
        )

    return mod_tiles


def get_slices(
    tile_shape: tuple, overlap: int, y_tile_id: int, x_tile_id: int, y_id_max: int, x_id_max: int
) -> Tuple[Tuple[slice, slice], Tuple[slice, slice]]:
    if y_id_max - 1 == 0:
        tile_slice_y = slice(overlap, tile_shape[0] + overlap)
        y_f = 0
        y_t = tile_shape[0]
    elif y_tile_id == 0:
        tile_slice_y = slice(overlap, tile_shape[0] + overlap * 2)
        y_f = 0
        y_t = tile_shape[0] + overlap
    elif y_tile_id == y_id_max - 1:
        tile_slice_y = slice(overlap, tile_shape[0] + overlap)
        y_f = y_tile_id * tile_shape[0]
        y_t = y_f + tile_shape[0]
    else:
        tile_slice_y = slice(overlap, tile_shape[0] + overlap * 2)
        y_f = y_tile_id * tile_shape[0]
        y_t = y_f + tile_shape[0] + overlap

    if x_id_max - 1 == 0:
        tile_slice_x = slice(overlap, tile_shape[1] + overlap)
        x_f = 0
        x_t = tile_shape[1]
    elif x_tile_id == 0:
        tile_slice_x = slice(overlap, tile_shape[1] + overlap * 2)
        x_f = 0
        x_t = tile_shape[1] + overlap
    elif x_tile_id == x_id_max - 1:
        tile_slice_x = slice(overlap, tile_shape[1] + overlap)
        x_f = x_tile_id * tile_shape[1]
        x_t = x_f + tile_shape[1]
    else:
        tile_slice_x = slice(overlap, tile_shape[1] + overlap * 2)
        x_f = x_tile_id * tile_shape[1]
        x_t = x_f + tile_shape[1] + overlap

    tile_slice = (tile_slice_y, tile_slice_x)
    big_image_slice = (slice(y_f, y_t), slice(x_f, x_t))

    return tile_slice, big_image_slice


def stitch_mask(
    tiles: List[Image],
    y_ntiles: int,
    x_ntiles: int,
    tile_shape: list,
    dtype,
    overlap: int,
    padding: dict,
) -> Image:

    y_axis = -2
    x_axis = -1

    tile_y_size = tile_shape[y_axis] - overlap * 2
    tile_x_size = tile_shape[x_axis] - overlap * 2

    big_image_y_size = y_ntiles * tile_y_size
    big_image_x_size = x_ntiles * tile_x_size

    y_pad = padding["top"] + padding["bottom"]
    x_pad = padding["left"] + padding["right"]

    big_image_shape = (big_image_y_size, big_image_x_size)
    big_image = np.zeros(big_image_shape, dtype=dtype)

    print("n tiles x,y:", (x_ntiles, y_ntiles))
    print("plane shape x,y:", big_image_x_size - x_pad, big_image_y_size - y_pad)

    n = 0
    for i in range(0, y_ntiles):
        for j in range(0, x_ntiles):
            tile_slice, big_image_slice = get_slices(
                (tile_y_size, tile_x_size), overlap, i, j, y_ntiles, x_ntiles
            )

            tile = tiles[n]
            tile = tile.astype(dtype)

            mask_zeros = big_image[big_image_slice] == 0
            big_image[big_image_slice][mask_zeros] = tile[tile_slice][mask_zeros]

            n += 1

    new_big_image_shape = (big_image_shape[0] - y_pad, big_image_shape[1] - x_pad)
    return big_image[: new_big_image_shape[0], : new_big_image_shape[1]]
