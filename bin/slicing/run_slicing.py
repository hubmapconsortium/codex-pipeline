import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import tifffile as tif
from modify_pipeline_config import modify_pipeline_config, save_modified_pipeline_config
from slicer import slice_img


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


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_image_path_in_dir(dir_path: Path) -> Path:
    allowed_extensions = (".tif", ".tiff")
    listing = list(dir_path.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    return img_listing[0]


def get_stitched_image_shape(
    stitched_dirs: Dict[int, Dict[int, Dict[int, Path]]],
) -> Tuple[int, int]:
    for cycle in stitched_dirs:
        for region in stitched_dirs[cycle]:
            for channel, dir_path in stitched_dirs[cycle][region].items():
                stitched_img_path = get_image_path_in_dir(dir_path)
                break
    with tif.TiffFile(stitched_img_path) as TF:
        stitched_image_shape = TF.series[0].shape
    return stitched_image_shape


def create_output_dirs_for_tiles(
    stitched_channel_dirs: Dict[int, Dict[int, Dict[int, Path]]], out_dir: Path
) -> Dict[int, Dict[int, Path]]:
    dir_naming_template = "Cyc{cycle:d}_reg{region:d}"
    out_dirs_for_tiles = dict()
    for cycle in stitched_channel_dirs:
        out_dirs_for_tiles[cycle] = {}
        for region in stitched_channel_dirs[cycle]:
            out_dir_name = dir_naming_template.format(cycle=cycle, region=region)
            out_dir_path = out_dir / out_dir_name
            make_dir_if_not_exists(out_dir_path)
            out_dirs_for_tiles[cycle][region] = out_dir_path
    return out_dirs_for_tiles


def split_channels_into_tiles(
    stitched_dirs: Dict[int, Dict[int, Dict[int, Path]]],
    out_dirs_for_tiles: Dict[int, Dict[int, Path]],
    tile_size=1000,
    overlap=50,
):
    for cycle in stitched_dirs:
        for region in stitched_dirs[cycle]:
            for channel, dir_path in stitched_dirs[cycle][region].items():
                stitched_image_path = get_image_path_in_dir(dir_path)
                print(stitched_image_path.name)
                out_dir = out_dirs_for_tiles[cycle][region]
                slice_img(
                    path_to_str(stitched_image_path),
                    path_to_str(out_dir),
                    tile_size=tile_size,
                    overlap=overlap,
                    region=region,
                    zplane=1,
                    channel=channel,
                )


def organize_dirs(base_stitched_dir: Path) -> Dict[int, Dict[int, Dict[int, Path]]]:
    stitched_channel_dirs = list(base_stitched_dir.iterdir())
    # expected dir naming Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}
    stitched_dirs = dict()
    for dir_path in stitched_channel_dirs:
        name_info = path_to_dict(dir_path)
        cycle = name_info["Cyc"]
        region = name_info["Reg"]
        channel = name_info["Ch"]

        if cycle in stitched_dirs:
            if region in stitched_dirs[cycle]:
                stitched_dirs[cycle][region][channel] = dir_path
            else:
                stitched_dirs[cycle][region] = {channel: dir_path}
        else:
            stitched_dirs[cycle] = {region: {channel: dir_path}}
    return stitched_dirs


def main(base_stitched_dir: Path, pipeline_config_path: Path):
    out_dir = Path("/output/new_tiles")
    pipeline_conf_dir = Path("/output/pipeline_conf/")
    make_dir_if_not_exists(out_dir)
    make_dir_if_not_exists(pipeline_conf_dir)

    stitched_channel_dirs = organize_dirs(base_stitched_dir)
    out_dirs_for_tiles = create_output_dirs_for_tiles(stitched_channel_dirs, out_dir)

    stitched_img_shape = get_stitched_image_shape(stitched_channel_dirs)

    tile_size = 1000
    overlap = 100
    print("Splitting images into tiles")
    print("Tile size:", tile_size, "| overlap:", overlap)
    split_channels_into_tiles(stitched_channel_dirs, out_dirs_for_tiles, tile_size, overlap)

    modified_experiment = modify_pipeline_config(
        pipeline_config_path, (tile_size, tile_size), overlap, stitched_img_shape
    )
    save_modified_pipeline_config(modified_experiment, pipeline_conf_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_stitched_dir",
        type=Path,
        help="path to directory with directories per channel that contain stitched images",
    )
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )

    args = parser.parse_args()

    main(args.base_stitched_dir, args.pipeline_config_path)
