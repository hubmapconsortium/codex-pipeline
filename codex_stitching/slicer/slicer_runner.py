from pathlib import Path

from slicer.slicer import slice_img


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def get_image_path_in_dir(dir_path: Path) -> Path:
    allowed_extensions = (".tif", ".tiff")
    listing = list(dir_path.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    return img_listing[0]


def split_channels_into_tiles(
    stitched_dirs: dict, new_tiles_per_cycle_region: dict, tile_size=1000, overlap=50
):
    for cycle in stitched_dirs:
        for region in stitched_dirs[cycle]:
            for channel, dir_path in stitched_dirs[cycle][region].items():
                stitched_image_path = get_image_path_in_dir(dir_path)
                out_dir = new_tiles_per_cycle_region[cycle][region]
                slice_img(
                    path_to_str(stitched_image_path),
                    out_dir,
                    tile_size=tile_size,
                    overlap=overlap,
                    region=region,
                    zplane=1,
                    channel=channel,
                )
