import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import czifile
import numpy as np
from tifffile import imwrite

# requires
# numpy
# czifile
# imagecodecs
# tifffile


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def read_and_write_tile(
    sub_block_directory: czifile.DirectoryEntryDV,
) -> None:
    """
    mapped function to read tile metadata and data and write to disk
    Parameters
    ----------
    sub_block_directory: czifile.DirectoryEntryDV
        czi data sub-block
    """
    tile_name_template = "{region:d}_{tile:05d}_Z{zplane:03d}_CH{channel:d}.tif"

    # get c and z indices
    c_axis_idx = sub_block_directory.axes.index("C")
    z_axis_idx = sub_block_directory.axes.index("Z")

    # get c and z axis position for data sub-block
    c_dim_entry = sub_block_directory.dimension_entries[c_axis_idx + 1]
    z_dim_entry = sub_block_directory.dimension_entries[z_axis_idx + 1]

    c_idx = c_dim_entry.start
    z_idx = z_dim_entry.start
    tile_idx = sub_block_directory.mosaic_index

    # read data into python
    data_seg = sub_block_directory.data_segment()
    im = data_seg.data()

    # construct name
    tile_out_name = tile_name_template.format(
        region=region_no_in, tile=tile_idx + 1, zplane=z_idx + 1, channel=c_idx + 1
    )

    output_path = Path(output_dir_in) / tile_out_name
    # write data
    imwrite(path_to_str(output_path), np.squeeze(im))


def convert_czi_to_tiles(
    czi_fp: Union[str, Path],
    region_no: int,
    output_dir: Union[str, Path],
    max_workers: Optional[int] = None,
) -> None:
    """
    Reads a CODEX zeiss .czi and dumps the tiles onto disk mimicing akoya layout
    Parameters
    ----------
    czi_fp : Union[str, Path]
        file path to the Zeiss .czi file
    region_no : int
        region number of the CODEX run
    output_dir : Union[str, Path]
        directory where tiles will be saved
    max_workers: Optional[int]
        how many multiprocessing works to use. If none, it will detect number of
        cores and use all
    """
    print("Converting", str(czi_fp))

    czi = czifile.CziFile(czi_fp)
    sbs = czi.filtered_subblock_directory

    tile_indices = [s.mosaic_index for s in sbs]
    tile_indices, tile_counts = np.unique(tile_indices, return_counts=True)

    if np.all(tile_counts) is False:
        raise ValueError("no all tiles contain the same number of c,z,y,x planes")

    # make sure these variables are in scope for the ThreadPoolExecutor
    # mapping an iterator with some other variables is a pain
    global region_no_in
    global output_dir_in
    region_no_in = region_no
    output_dir_in = output_dir

    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1

    if max_workers > 1:
        czi._fh.lock = True
        with ThreadPoolExecutor(max_workers) as executor:
            executor.map(
                read_and_write_tile,
                czi.filtered_subblock_directory,
            )
        czi._fh.lock = None
    else:
        for directory_entry in czi.filtered_subblock_directory:
            read_and_write_tile(directory_entry)
    print("Finished")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    start = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--czi_path", type=Path, help="path to the czi file to convert")
    parser.add_argument(
        "--region_no",
        type=int,
        help="region number of the czi cycles (this should always be 1, "
        "but this argument is futureproofing)",
    )
    parser.add_argument("--output_dir", type=Path, help="directory to place to the tiles")
    parser.add_argument("--max_workers", nargs="?", const=None, type=int)

    args = parser.parse_args()

    convert_czi_to_tiles(
        args.czi_path, args.region_no, args.output_dir, max_workers=args.max_workers
    )
    print("time elapsed", str(datetime.now() - start))
