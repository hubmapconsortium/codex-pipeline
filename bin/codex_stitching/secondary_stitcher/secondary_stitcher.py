import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import tifffile as tif
from mask_stitching import process_all_masks
from skimage.measure import regionprops_table

Image = np.ndarray


def add_structured_annotations(omexml_str: str, nucleus_channel: str, cell_channel: str) -> str:
    """
    Will add this, to the root, after Image node
    <StructuredAnnotations>
    <XMLAnnotation ID="Annotation:0">
        <Value>
            <OriginalMetadata>
                <Key>SegmentationChannels</Key>
                <Value>
                    <Nucleus>DAPI-02</Nucleus>
                    <Cell>CD45</Cell>
                </Value>
            </OriginalMetadata>
        </Value>
    </XMLAnnotation>
    </StructuredAnnotations>
    """

    # Remove some prefixes
    nucleus_channel = re.sub(r"cyc(\d+)_ch(\d+)_orig(.*)", r"\3", nucleus_channel)
    cell_channel = re.sub(r"cyc(\d+)_ch(\d+)_orig(.*)", r"\3", cell_channel)

    structured_annotation = ET.Element("StructuredAnnotations")
    annotation = ET.SubElement(structured_annotation, "XMLAnnotation", {"ID": "Annotation:0"})
    annotation_value = ET.SubElement(annotation, "Value")
    original_metadata = ET.SubElement(annotation_value, "OriginalMetadata")
    segmentation_channels_key = ET.SubElement(
        original_metadata, "Key"
    ).text = "SegmentationChannels"
    segmentation_channels_value = ET.SubElement(original_metadata, "Value")
    ET.SubElement(segmentation_channels_value, "Nucleus").text = nucleus_channel
    ET.SubElement(segmentation_channels_value, "Cell").text = cell_channel
    sa_str = ET.tostring(structured_annotation, encoding="utf-8").decode("utf-8")

    if "StructuredAnnotations" in omexml_str:
        sa_placement = omexml_str.find("<StructuredAnnotations>") + len("<StructuredAnnotations>")
        sa_str = re.sub(r"</?StructuredAnnotations>", "", sa_str)
    else:
        sa_placement = omexml_str.find("</Image>") + len("</Image>")

    omexml_str_with_sa = omexml_str[:sa_placement] + sa_str + omexml_str[sa_placement:]
    return omexml_str_with_sa


def alpha_num_order(string: str) -> str:
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def get_img_listing(in_dir: Path) -> List[Path]:
    allowed_extensions = (".tif", ".tiff")
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
    value_list = re.split(r"(\d+)(?:_?)", path.name)[:-1]
    d = dict(zip(*[iter(value_list)] * 2))
    d = {k: int(v) for k, v in d.items()}
    d.update({"path": path})
    return d


def get_slices(
    arr: np.ndarray, hor_f: int, hor_t: int, ver_f: int, ver_t: int, padding: dict, overlap=0
):
    left_check = hor_f - padding["left"]
    top_check = ver_f - padding["top"]
    right_check = hor_t - arr.shape[-1]
    bot_check = ver_t - arr.shape[-2]

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
    tile_slice = (
        slice(top_pad_size + overlap, tile_shape[0] + overlap),
        slice(left_pad_size + overlap, tile_shape[1] + overlap),
    )

    return big_image_slice, tile_slice


def get_dataset_info(img_dir: Path):
    img_paths = get_img_listing(img_dir)
    positions = [path_to_dict(p) for p in img_paths]
    df = pd.DataFrame(positions)
    df.sort_values(["R", "Y", "X"], inplace=True)
    df.reset_index(inplace=True)

    region_ids = list(df["R"].unique())
    y_ntiles = df["Y"].max()
    x_ntiles = df["X"].max()

    path_list_per_region = []

    for r in region_ids:
        region_selection = df[df["R"] == r].index
        path_list = list(df.loc[region_selection, "path"])
        path_list_per_region.append(path_list)

    return path_list_per_region, y_ntiles, x_ntiles


def load_tiles(path_list: List[Path], key: Union[None, int]):
    tiles = []
    if key is None:
        for path in path_list:
            tiles.append(tif.imread(path_to_str(path)))
    else:
        for path in path_list:
            tiles.append(tif.imread(path_to_str(path), key=key))

    return tiles


def calc_mask_coverage(segm_mask: Image) -> float:
    mask_pixels = np.sum(segm_mask != 0)
    total_pixels = segm_mask.shape[-2] * segm_mask.shape[-1]
    return float(round(mask_pixels / total_pixels, 3))


def calc_snr(img: Image) -> float:
    return float(round(np.mean(img) / np.std(img), 3))


def calc_label_sizes(segm_mask: Image) -> Dict[str, List[float]]:
    # bounding boxes around labels
    # useful to check if there are merged labels
    props = regionprops_table(segm_mask, properties=("label", "bbox"))
    min_rows = props["bbox-0"]
    min_cols = props["bbox-1"]
    max_rows = props["bbox-2"]
    max_cols = props["bbox-3"]
    bbox_arr = np.stack((min_rows, max_rows, min_cols, max_cols), axis=1)
    dif = np.stack((bbox_arr[:, 1] - bbox_arr[:, 0], bbox_arr[:, 3] - bbox_arr[:, 2]), axis=1)
    long_sides = np.max(dif, axis=1)
    label_sizes = dict(
        min_bbox_size=[float(i) for i in dif[np.argmin(long_sides)].tolist()],
        max_bbox_size=[float(i) for i in dif[np.argmax(long_sides)].tolist()],
        mean_bbox_size=[float(i) for i in np.round(np.mean(dif, axis=0), 3).tolist()],
    )
    return label_sizes


def stitch_plane(
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

    big_image_y_size = (y_ntiles * tile_y_size) - padding["top"] - padding["bottom"]
    big_image_x_size = (x_ntiles * tile_x_size) - padding["left"] - padding["right"]

    big_image_shape = (big_image_y_size, big_image_x_size)
    big_image = np.zeros(big_image_shape, dtype=dtype)

    print("n tiles x,y:", (x_ntiles, y_ntiles))
    print("plane shape x,y:", big_image_shape[::-1])
    n = 0
    for i in range(0, y_ntiles):
        ver_f = i * tile_y_size
        ver_t = ver_f + tile_y_size

        for j in range(0, x_ntiles):
            hor_f = j * tile_x_size
            hor_t = hor_f + tile_x_size

            big_image_slice, tile_slice = get_slices(
                big_image, hor_f, hor_t, ver_f, ver_t, padding, overlap
            )
            tile = tiles[n]

            big_image[tuple(big_image_slice)] = tile[tuple(tile_slice)]

            n += 1
    return big_image


def main(
    img_dir: Path,
    out_dir: Path,
    img_name_template: str,
    overlap: int,
    padding_str: str,
    is_mask: bool,
    nucleus_channel: str,
    cell_channel: str,
):
    padding_int = [int(i) for i in padding_str.split(",")]
    padding = {
        "left": padding_int[0],
        "right": padding_int[1],
        "top": padding_int[2],
        "bottom": padding_int[3],
    }

    path_list_per_region, y_ntiles, x_ntiles = get_dataset_info(img_dir)

    with tif.TiffFile(path_to_str(path_list_per_region[0][0])) as TF:
        tile_shape = list(TF.series[0].shape)
        npages = len(TF.pages)
        dtype = TF.series[0].dtype
        ome_meta = TF.ome_metadata

    big_image_y_size = (
        (y_ntiles * (tile_shape[-2] - overlap * 2)) - padding["top"] - padding["bottom"]
    )
    big_image_x_size = (
        (x_ntiles * (tile_shape[-1] - overlap * 2)) - padding["left"] - padding["right"]
    )

    if is_mask:
        dtype = np.uint32
    else:
        ome_meta = re.sub(r'\sSizeY="\d+"', ' SizeY="' + str(big_image_y_size) + '"', ome_meta)
        ome_meta = re.sub(r'\sSizeX="\d+"', ' SizeX="' + str(big_image_x_size) + '"', ome_meta)
        ome_meta = re.sub(r'\sDimensionOrder="[XYCZT]+"', ' DimensionOrder="XYZCT"', ome_meta)
        ome_meta = add_structured_annotations(ome_meta, nucleus_channel, cell_channel)
    # part of this report is generated after mask stitching and part after expression stitching

    total_report = dict()
    for r, path_list in enumerate(path_list_per_region):
        new_path = out_dir / img_name_template.format(r=r + 1)
        this_region_report = dict()
        TW = tif.TiffWriter(path_to_str(new_path), bigtiff=True, shaped=False)
        if is_mask:
            # mask channels 0 - cells, 1 - nuclei, 2 - cell boundaries, 3 - nucleus boundaries
            tiles = load_tiles(path_list, key=None)
            masks, ome_meta = process_all_masks(
                tiles, tile_shape, y_ntiles, x_ntiles, overlap, padding, dtype
            )
            for mask in masks:
                new_shape = (1, mask.shape[0], mask.shape[1])
                TW.write(
                    mask.reshape(new_shape),
                    contiguous=True,
                    photometric="minisblack",
                    description=ome_meta,
                )

            this_region_report["num_cells"] = int(masks[0].max())
            this_region_report["num_nuclei"] = int(masks[1].max())
            this_region_report["cell_coverage"] = calc_mask_coverage(masks[0])
            this_region_report["nuclei_coverage"] = calc_mask_coverage(masks[1])
            this_region_report["cell_sizes"] = calc_label_sizes(masks[0])
            this_region_report["nucleus_sizes"] = calc_label_sizes(masks[1])
        else:
            for p in range(0, npages):
                tiles = load_tiles(path_list, key=p)
                print("\nstitching expressions page", p + 1, "/", npages)
                plane = stitch_plane(
                    tiles, y_ntiles, x_ntiles, tile_shape, dtype, overlap, padding
                )
                new_shape = (1, plane.shape[0], plane.shape[1])
                if p == 0:
                    this_region_report["num_channels"] = int(npages)
                    this_region_report["img_height"] = int(plane.shape[0])
                    this_region_report["img_width"] = int(plane.shape[1])
                    this_region_report["per_channel_snr"] = dict()
                    this_region_report["nucleus_channel"] = nucleus_channel
                    this_region_report["cell_channel"] = cell_channel
                this_region_report["per_channel_snr"][p] = calc_snr(plane)
                TW.write(
                    plane.reshape(new_shape),
                    contiguous=True,
                    photometric="minisblack",
                    description=ome_meta,
                )
        total_report["reg" + str(r + 1)] = this_region_report
        TW.close()
    return total_report
