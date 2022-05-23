import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile as tif
from bundle_images import bundle_channel_images
from feature_reg import reg
from opt_flow_reg import opt_flow_reg

sys.path.append("/opt/")
sys.path.append("/opt/image_registration")

from pipeline_utils.pipeline_config_reader import load_dataset_info


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


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


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}


def organize_dirs(base_stitched_dir: Path) -> Dict[int, Dict[int, Dict[int, Path]]]:
    stitched_channel_dirs = list(base_stitched_dir.iterdir())
    # expected dir naming Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}
    stitched_dirs = dict()
    for dir_path in stitched_channel_dirs:
        name_info = path_to_dict(dir_path)
        region = name_info["Reg"]
        cycle = name_info["Cyc"]
        channel = name_info["Ch"]

        if region in stitched_dirs:
            if cycle in stitched_dirs[region]:
                stitched_dirs[region][cycle][channel] = dir_path
            else:
                stitched_dirs[region][cycle] = {channel: dir_path}
        else:
            stitched_dirs[region] = {cycle: {channel: dir_path}}
    stitched_dirs = sort_dict(stitched_dirs)
    return stitched_dirs


def save_to_stack(path_list, out_path, ome_meta):
    imgs = []
    for path in path_list:
        imgs.append(tif.imread(path))
    stack = np.stack(imgs)
    TW = tif.TiffWriter(out_path, bigtiff=True)
    TW.write(stack, contiguous=True, description=ome_meta)
    TW.close()


def set_channel_names_and_paths(
    channels_per_cycle: Dict[int, Dict[int, Path]], channel_names: List[str]
):
    n = 0
    new_channels_per_cycle = dict()
    for cycle in channels_per_cycle:
        for channel, channel_path in channels_per_cycle[cycle].items():
            channel_name = channel_names[n]
            if cycle in new_channels_per_cycle:
                new_channels_per_cycle[cycle][channel_name] = channel_path / "fused_tp_0_ch_0.tif"
            else:
                new_channels_per_cycle[cycle] = {
                    channel_name: channel_path / "fused_tp_0_ch_0.tif"
                }
            n += 1
    return new_channels_per_cycle


def save_images_to_stacks(
    stitched_dirs: Dict[int, Dict[int, Dict[int, Path]]], out_dir: Path, channel_names: List[str]
) -> Dict[int, List[Path]]:
    region_dir_t = "region_{reg:03d}"
    cycle_stacks_per_region = dict()
    for region in stitched_dirs:
        region_out_dir = out_dir / region_dir_t.format(reg=region)
        channels_per_cycle = stitched_dirs[region]
        new_channels_per_cycle = set_channel_names_and_paths(channels_per_cycle, channel_names)
        cycle_paths = bundle_channel_images(new_channels_per_cycle, region_out_dir, "separate")
        cycle_stacks_per_region[region] = cycle_paths
    return cycle_stacks_per_region


def run_feature_reg(img_paths, ref_img_id, ref_channel, out_dir, n_workers):
    img_paths2 = [path_to_str(p) for p in img_paths]
    reg.main(
        img_paths2,
        ref_img_id,
        ref_channel,
        path_to_str(out_dir),
        n_workers,
        tile_size=1000,
        num_pyr_lvl=3,
        num_iter=3,
        stack=False,
        estimate_only=False,
        load_param="none",
    )


def run_opt_flow_reg(stack_path, ref_channel, out_dir, n_workers):
    opt_flow_reg.main(
        path_to_str(stack_path),
        ref_channel,
        path_to_str(out_dir),
        n_workers,
        tile_size=1000,
        overlap=100,
        num_pyr_lvl=3,
        num_iter=3,
    )


def split_to_cycle_channels(stack_path, out_dir, num_cycles, num_channels):
    dir_name_t = "Cyc{cyc:03d}_Reg{reg:03d}_Ch{ch:03d}"
    stack = tif.imread(stack_path)
    print(stack.shape)
    n = 0
    print(num_cycles, num_channels)
    for cyc in range(1, num_cycles + 1):
        for ch in range(1, num_channels + 1):
            out_dir_per_channel = out_dir / dir_name_t.format(cyc=cyc, reg=1, ch=ch)
            make_dir_if_not_exists(out_dir_per_channel)
            tif.imwrite(out_dir_per_channel / "fused_tp_0_ch_0.tif", stack[n, :, :])
            n += 1
    return


def main(base_stitched_dir: Path, pipeline_config_path: Path):
    start = datetime.now()
    out_dir = Path("/output")
    reg_results_dir = out_dir / Path("registration_results")
    stack_dir = out_dir / Path("img_stacks")
    final_output = out_dir / Path("registered_images")

    make_dir_if_not_exists(out_dir)
    make_dir_if_not_exists(reg_results_dir)
    make_dir_if_not_exists(stack_dir)
    make_dir_if_not_exists(final_output)

    pipeline_config = load_dataset_info(pipeline_config_path)

    channel_names = pipeline_config["channel_names"]
    ref_cycle_id = pipeline_config["reference_cycle"] - 1  # from 1 based to 0 based
    ref_channel_name = channel_names[
        pipeline_config["reference_channel"] - 1
    ]  # from 1 based to 0 based
    num_cycles = pipeline_config["num_cycles"]
    num_channels = pipeline_config["num_channels"]
    n_workers = pipeline_config["num_concurrent_tasks"]

    stitched_dirs = organize_dirs(base_stitched_dir)
    cycle_stacks_per_region = save_images_to_stacks(stitched_dirs, stack_dir, channel_names)
    region_dir_t = "region_{reg:03d}"
    for region, stack_paths in cycle_stacks_per_region.items():
        region_dir = reg_results_dir / region_dir_t.format(reg=region)
        make_dir_if_not_exists(region_dir)

        run_feature_reg(stack_paths, ref_cycle_id, ref_channel_name, region_dir, n_workers)
        feature_reg_result = region_dir / "out.tif"
        run_opt_flow_reg(feature_reg_result, ref_channel_name, region_dir, n_workers)
        opt_flow_reg_result = region_dir / "out_opt_flow_registered.tif"
        split_to_cycle_channels(opt_flow_reg_result, final_output, num_cycles, num_channels)
    fin = datetime.now()
    print("Total time for registration", str(fin - start))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--pipeline_config", type=Path)

    args = parser.parse_args()

    main(args.data_dir, args.pipeline_config)
