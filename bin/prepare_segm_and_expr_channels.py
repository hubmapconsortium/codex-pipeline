import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import dask
import numpy as np
import tifffile as tif

sys.path.append("/opt/")
from ome_meta import OMEMetaCreator
from pipeline_utils.pipeline_config_reader import load_dataset_info


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def create_dirs_per_region(
    listing: Dict[int, Dict[int, Dict[int, Path]]], out_dir: Path
) -> Dict[int, Path]:
    dirs_per_region = dict()
    dir_name_template = "region_{region:03d}"
    for region in listing:
        dir_path = out_dir / dir_name_template.format(region=region)
        make_dir_if_not_exists(dir_path)
        dirs_per_region[region] = dir_path
    return dirs_per_region


def copy_segm_channels_to_out_dirs(
    segm_ch_paths: Dict[int, Dict[str, Path]],
    dirs_per_region: Dict[int, Path],
):
    print("Preparing segmentation channels")
    tasks = []
    new_name_t = "reg{region:03d}_{segm_ch_type}.tif"
    for region in segm_ch_paths:
        nuc_path = segm_ch_paths[region]["nucleus"]
        cell_path = segm_ch_paths[region]["cell"]

        nuc_ch_name = new_name_t.format(region=region, segm_ch_type="nucleus")
        cell_ch_name = new_name_t.format(region=region, segm_ch_type="cell")

        reg_out_dir = dirs_per_region[region]
        nuc_out_path = reg_out_dir / nuc_ch_name
        cell_out_path = reg_out_dir / cell_ch_name

        print("region", str(region), "| src:", str(nuc_path), "| dst:", str(nuc_out_path))
        print("region", str(region), "| src:", str(cell_path), "| dst:", str(cell_out_path))

        tasks.append(dask.delayed(shutil.copy)(nuc_path, nuc_out_path))
        tasks.append(dask.delayed(shutil.copy)(cell_path, cell_out_path))
    dask.compute(*tasks)


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


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}


def get_img_dirs(base_img_dir: Path) -> Dict[int, Dict[int, Dict[int, Path]]]:
    stitched_channel_dirs = list(base_img_dir.iterdir())
    # expected dir naming Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}
    img_dirs = dict()
    for dir_path in stitched_channel_dirs:
        name_info = path_to_dict(dir_path)
        region = name_info["Reg"]
        cycle = name_info["Cyc"]
        channel = name_info["Ch"]

        if region in img_dirs:
            if cycle in img_dirs[region]:
                img_dirs[region][cycle][channel] = dir_path
            else:
                img_dirs[region][cycle] = {channel: dir_path}
        else:
            img_dirs[region] = {cycle: {channel: dir_path}}
    img_dirs = sort_dict(img_dirs)
    return img_dirs


def get_path_to_segm_channels(
    listing: Dict[int, Dict[int, Dict[int, Path]]],
    nuc_ch_loc: dict,
    membr_ch_loc: dict,
) -> Dict[int, Dict[str, Path]]:
    per_reg_seg_ch_paths = dict()
    for region in listing:
        nuc_img_path = listing[region][nuc_ch_loc["CycleID"]][nuc_ch_loc["ChannelID"]]
        cell_img_path = listing[region][membr_ch_loc["CycleID"]][membr_ch_loc["ChannelID"]]
        segm_ch_paths = {"nucleus": nuc_img_path, "cell": cell_img_path}
        per_reg_seg_ch_paths[region] = segm_ch_paths
    return per_reg_seg_ch_paths


def get_img_listing(
    img_dir_listing: Dict[int, Dict[int, Dict[int, Path]]], img_name: str
) -> Dict[int, Dict[int, Dict[int, Path]]]:
    img_listing = dict()
    for region in img_dir_listing:
        img_listing[region] = dict()
        n = 0
        for cycle in img_dir_listing[region]:
            img_listing[region][cycle] = dict()
            for ch, ch_dir_path in img_dir_listing[region][cycle].items():
                img_listing[region][cycle][ch] = ch_dir_path / img_name
    return img_listing


def check_if_needed(
    channel_name: str, channel_names_qc_pass: Dict[str, List[str]], nuclei_channel: str
) -> bool:
    if channel_name == nuclei_channel:
        return True
    if channel_names_qc_pass[channel_name] == ["TRUE"]:
        pass
    else:
        return False
    unwanted_ch_patterns = [r"^blank", r"^empty", r"^DAPI", r"^HOECHST"]
    for unwanted_ch_pat in unwanted_ch_patterns:
        if re.match(unwanted_ch_pat, channel_name, re.IGNORECASE):
            return False
    else:
        return True


def filter_expr_channels(
    listing: Dict[int, Dict[int, Dict[int, Path]]],
    channel_names: List[str],
    channel_names_qc_pass: Dict[str, List[str]],
    nuclei_channel: str,
) -> Tuple[Dict[int, Dict[int, Dict[int, Path]]], List[str]]:
    preserved_channels = []
    filtered_expr_ch_listing = dict()
    for region in listing:
        filtered_expr_ch_listing[region] = dict()
        n = 0
        for cycle in listing[region]:
            channel_dict = dict()
            for ch, ch_path in listing[region][cycle].items():
                ch_name = channel_names[n]
                is_needed = check_if_needed(ch_name, channel_names_qc_pass, nuclei_channel)
                if is_needed:
                    channel_dict[ch] = ch_path
                    preserved_channels.append(ch_name)
                n += 1
            if len(channel_dict) != 0:
                filtered_expr_ch_listing[region][cycle] = channel_dict
    if preserved_channels == []:
        msg = "All channels have been filtered as unwanted. Check the metadata!"
        raise ValueError(msg)
    return filtered_expr_ch_listing, preserved_channels


def write_expr_channels(
    out_dir: Path,
    out_img_name_t: str,
    expr_ch_paths: Dict[int, Dict[int, Dict[int, Path]]],
    ome_meta_per_region: Dict[int, str],
):
    print("Writing expression channels into stack per region")
    for region in expr_ch_paths:
        out_img_name = out_img_name_t.format(region=region)

        img_list = []
        for cycle in expr_ch_paths[region]:
            for ch, ch_path in expr_ch_paths[region][cycle].items():
                img = tif.imread(ch_path).squeeze()
                img_list.append(img)

        img_stack = np.stack(img_list, axis=0)  # c,y,x
        del img_list, img
        img_stack = np.expand_dims(img_stack, axis=1)  # c,z,y,x
        ome_meta = ome_meta_per_region[region]
        print(
            "region",
            str(region),
            "| src:",
            [str(expr_ch_paths[region])],
            "| dst:",
            str(Path(out_dir / out_img_name)),
        )
        with tif.TiffWriter(out_dir / out_img_name, bigtiff=True) as TW:
            TW.write(img_stack, contiguous=True, photometric="minisblack", description=ome_meta)


def generate_ome_meta_per_region(
    listing: Dict[int, Dict[int, Dict[int, Path]]],
    pipeline_config: dict,
    nuclei_channel: str,
    membrane_channel: str,
    channel_list: List[str],
) -> Dict[int, str]:
    ome_meta_per_region = dict()
    for region in listing:
        first_cycle = list(listing[region].keys())[0]
        first_channel = list(listing[region][first_cycle])[0]
        first_channel_path = listing[region][first_cycle][first_channel]

        meta_creator = OMEMetaCreator()
        meta_creator.pipeline_config = pipeline_config
        meta_creator.nucleus_channel = nuclei_channel
        meta_creator.cell_channel = membrane_channel
        meta_creator.channel_names = channel_list
        meta_creator.path_to_sample_img = first_channel_path
        ome_meta = meta_creator.create_ome_meta()
        ome_meta_per_region[region] = ome_meta
    return ome_meta_per_region


def main(data_dir: Path, pipeline_config_path: Path):
    # img_dir_name_t = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}"
    img_name = "fused_tp_0_ch_0.tif"
    expr_out_img_name_t = "reg{region:03d}_expr.ome.tiff"

    out_dir = Path("/output")
    segm_ch_out_dir = out_dir / "segmentation_channels"
    expr_ch_out_dir = out_dir / "expression_channels"
    make_dir_if_not_exists(segm_ch_out_dir)
    make_dir_if_not_exists(expr_ch_out_dir)

    pipeline_config = load_dataset_info(pipeline_config_path)

    nuclei_channel = pipeline_config["nuclei_channel"]
    membrane_channel = pipeline_config["membrane_channel"]

    nuclei_channel_location = pipeline_config["nuclei_channel_loc"]
    membrane_channel_location = pipeline_config["membrane_channel_loc"]
    channel_names = pipeline_config["channel_names"]
    channel_names_qc_pass = pipeline_config["channel_names_qc_pass"]

    img_dir_listing = get_img_dirs(data_dir)
    listing = get_img_listing(img_dir_listing, img_name)

    segm_ch_paths = get_path_to_segm_channels(
        listing, nuclei_channel_location, membrane_channel_location
    )
    expr_ch_paths, preserved_channels = filter_expr_channels(
        listing, channel_names, channel_names_qc_pass, nuclei_channel
    )

    ome_meta_per_region = generate_ome_meta_per_region(
        listing, pipeline_config, nuclei_channel, membrane_channel, preserved_channels
    )

    num_workers = pipeline_config["num_concurrent_tasks"]
    dask.config.set({"num_workers": num_workers, "scheduler": "processes"})

    segm_ch_dirs_per_region = create_dirs_per_region(listing, segm_ch_out_dir)
    copy_segm_channels_to_out_dirs(segm_ch_paths, segm_ch_dirs_per_region)

    write_expr_channels(expr_ch_out_dir, expr_out_img_name_t, expr_ch_paths, ome_meta_per_region)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to the dataset directory")
    parser.add_argument("--pipeline_config", type=Path, help="path to dataset metadata yaml")
    args = parser.parse_args()

    main(args.data_dir, args.pipeline_config)
