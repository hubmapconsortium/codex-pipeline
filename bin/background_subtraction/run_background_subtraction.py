import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import dask
import numpy as np
import tifffile as tif

sys.path.append("/opt/")
from pipeline_utils.dataset_listing import get_img_listing
from pipeline_utils.pipeline_config_reader import load_dataset_info

ImgStack = np.ndarray
Image = np.ndarray


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def organize_channels_per_cycle(
    channel_names: List[str], num_cycles: int, num_channels_per_cycle: int
) -> Dict[int, Dict[int, str]]:
    """{1: {1: 'DAPI-01', 2: 'Blank', 3: 'Blank', 4: 'Blank'},}"""
    channels_per_cycle = {cyc: dict() for cyc in range(1, num_cycles + 1)}
    cycle = 0
    ch_id = 1
    for i in range(0, len(channel_names)):
        if i % num_channels_per_cycle == 0:
            cycle += 1
            ch_id = 1
        channels_per_cycle[cycle][ch_id] = channel_names[i]
        ch_id += 1
    return channels_per_cycle


def filter_channels(
    channels_per_cycle: Dict[int, Dict[int, str]], channel_names_in_stack: List[str]
) -> Dict[int, Dict[int, str]]:
    filtered_ch_per_cycle = {cyc: dict() for cyc in channels_per_cycle.keys()}
    channel_names_in_stack_lookup = [ch.lower() for ch in channel_names_in_stack]
    for cycle in channels_per_cycle:
        for ch_id, channel_name in channels_per_cycle[cycle].items():
            if channel_name.lower() in channel_names_in_stack_lookup:
                filtered_ch_per_cycle[cycle][ch_id] = channel_name
    return filtered_ch_per_cycle


def get_channel_names_in_stack(img_path: Path) -> List[str]:
    with tif.TiffFile(str(img_path.absolute())) as TF:
        ij_meta = TF.imagej_metadata
    channel_names = [ch_name.lstrip("proc_").lower() for ch_name in ij_meta["Labels"]]
    return channel_names


def get_num_bg_ch_per_cycle(
    channels_per_cycle: Dict[int, List[str]], background_channel: str
) -> Dict[int, int]:
    num_bg_ch_per_cyc = {cyc: 0 for cyc in channels_per_cycle.keys()}
    bg_ch_pattern = re.compile(r"^" + background_channel, re.IGNORECASE)
    for cycle, channels in channels_per_cycle.items():
        for ch_name in channels:
            if bg_ch_pattern.match(ch_name):
                num_bg_ch_per_cyc[cycle] += 1
    return num_bg_ch_per_cyc


def get_stack_ids_of_bg_channels(
    channels_per_cycle: Dict[int, Dict[int, str]],
    background_channel: str,
) -> Dict[int, Dict[int, int]]:
    bg_channel_ids_per_cycle = {cyc: dict() for cyc in channels_per_cycle.keys()}
    n = 0
    bg_ch_pattern = re.compile(r"^" + background_channel, re.IGNORECASE)
    for cycle, channels in channels_per_cycle.items():
        for ch_id, ch_name in channels.items():
            if bg_ch_pattern.match(ch_name):
                bg_channel_ids_per_cycle[cycle][ch_id] = n
            n += 1
    return bg_channel_ids_per_cycle


def select_cycles_with_bg_ch(
    bg_channel_ids_per_cycle: Dict[int, Dict[int, int]], num_channels_per_cycle: int
) -> Dict[int, Dict[int, int]]:
    required_num_of_bg_channels = num_channels_per_cycle - 1
    selected_bg_channels_per_cyc = dict()
    for cycle, channels in bg_channel_ids_per_cycle.items():
        if len(channels.keys()) >= required_num_of_bg_channels:
            selected_bg_channels_per_cyc[cycle] = channels
    return selected_bg_channels_per_cyc


def get_stack_ids_for_bg_sub(
    channels_per_cycle: Dict[int, Dict[int, str]], bg_and_ref_nuc_ch: List[str]
):
    other_ex_ch = [ch.lower() for ch in bg_and_ref_nuc_ch]
    filtered_ch_per_cycle = {cyc: dict() for cyc in channels_per_cycle.keys()}
    n = 0
    for cycle, channels in channels_per_cycle.items():
        for ch_id, ch_name in channels.items():
            if ch_name.lower() not in other_ex_ch:
                filtered_ch_per_cycle[cycle][ch_id] = n
            n += 1
    return filtered_ch_per_cycle


def get_nuc_ch_stack_id(
    nuc_ch_name: str,
    channels_per_cycle: Dict[int, Dict[int, str]],
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
) -> int:
    for cycle in channels_per_cycle:
        for ch_id, ch_name in channels_per_cycle[cycle].items():
            if ch_name.lower() == nuc_ch_name.lower():
                nuc_ch_stack_id = stack_ids_per_cycle[cycle][ch_id]
                return nuc_ch_stack_id


def get_bg_ch_stack_ids(
    bg_ch_name: str,
    channels_per_cycle: Dict[int, Dict[int, str]],
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
) -> List[int]:
    bg_ch_stack_ids = []
    bg_ch_pattern = re.compile(r"^" + bg_ch_name, re.IGNORECASE)
    for cycle in channels_per_cycle:
        for ch_id, ch_name in channels_per_cycle[cycle].items():
            if bg_ch_pattern.match(ch_name):
                bg_ch_stack_ids.append(stack_ids_per_cycle[cycle][ch_id])
    return bg_ch_stack_ids


def assign_fraction_of_bg_mix_when_one_bg_cyc(
    cycles_with_bg_channels: Dict[int, Dict[int, int]], cycle_names: List[int]
) -> Dict[int, Dict[int, int]]:
    # {3: {1: 1},}
    cycles = sorted(cycle_names)
    bg_cycles = list(cycles_with_bg_channels.keys())
    distance_from_bg_cyc = abs(cycles.index(cycles[-1]) - cycles.index(bg_cycles[0]))
    fractions_per_cycle = dict()
    fractions = [1] * distance_from_bg_cyc
    cycle_subset = cycles[cycles.index(bg_cycles[0]) + 1 : cycles.index(cycles[-1]) + 1]
    fractions_per_cycle[bg_cycles[0]] = {bg_cycles[0]: {}}
    for i, cycle in enumerate(cycle_subset):
        fractions_per_cycle[cycle] = {bg_cycles[0]: fractions[i]}
    fractions_per_cycle_sorted = {
        k: v for k, v in sorted(fractions_per_cycle.items(), key=lambda item: item[0])
    }
    return fractions_per_cycle_sorted


def assign_fraction_of_bg_mix(
    cycles_with_bg_channels: Dict[int, Dict[int, int]], cycle_names: List[int]
) -> Dict[int, Dict[int, int]]:
    # {3: {1: 0.75, 9: 0.25},}
    cycles = sorted(cycle_names)
    bg_cycles = list(cycles_with_bg_channels.keys())
    distance_between_selected_cyc = abs(cycles.index(bg_cycles[1]) - cycles.index(bg_cycles[0]))

    fractions_per_cycle = dict()
    fractions = [
        round(i * (1 / distance_between_selected_cyc), 3)
        for i in range(1, distance_between_selected_cyc)
    ]
    cycle_subset = cycles[cycles.index(bg_cycles[0]) + 1 : cycles.index(bg_cycles[1])]

    fractions_per_cycle[bg_cycles[0]] = {bg_cycles[0]: {}, bg_cycles[1]: {}}
    fractions_per_cycle[bg_cycles[1]] = {bg_cycles[0]: {}, bg_cycles[1]: {}}

    for i, cycle in enumerate(cycle_subset):
        fractions_per_cycle[cycle] = {bg_cycles[0]: 1 - fractions[i], bg_cycles[1]: fractions[i]}

    fractions_per_cycle_sorted = {
        k: v for k, v in sorted(fractions_per_cycle.items(), key=lambda item: item[0])
    }
    return fractions_per_cycle_sorted


def create_new_channel_name_order(
    channels_per_cycle: Dict[int, Dict[int, str]], background_channel: str
) -> List[str]:
    channel_names = []
    bg_ch_pattern = re.compile(r"^" + background_channel, re.IGNORECASE)
    for cycle in channels_per_cycle:
        for ch_id, ch_name in channels_per_cycle[cycle].items():
            if bg_ch_pattern.match(ch_name):
                continue
            else:
                channel_names.append("proc_" + ch_name)
    return channel_names


def get_stack_ids_per_cycle(
    channels_per_cycle: Dict[int, Dict[int, str]], channel_names_in_stack: List[str]
) -> Dict[int, Dict[int, int]]:
    stack_ids_per_cycle = {cyc: dict() for cyc in channels_per_cycle.keys()}
    lookup_ch_name_list = [ch.lower() for ch in channel_names_in_stack]

    for cycle, channels in channels_per_cycle.items():
        for ch_id, ch_name in channels.items():
            if ch_name.lower() in lookup_ch_name_list:
                stack_id = lookup_ch_name_list.index(ch_name.lower())
                lookup_ch_name_list[stack_id] = None
                stack_ids_per_cycle[cycle][ch_id] = stack_id
    return stack_ids_per_cycle


def read_stack_and_meta(img_path: Path) -> Tuple[ImgStack, Dict[str, Any]]:
    with tif.TiffFile(str(img_path.absolute())) as TF:
        ij_meta = TF.imagej_metadata
        img = TF.series[0].asarray()
    return img, ij_meta


def save_stack(out_path: Path, img_stack: ImgStack, ij_meta: Dict[str, Any]):
    # output stack to will have more dimension to match TZCYX
    with tif.TiffWriter(str(out_path.absolute()), imagej=True) as TW:
        TW.write(
            np.expand_dims(img_stack, [0, 1]),
            contiguous=True,
            photometric="minisblack",
            metadata=ij_meta,
        )


def modify_initial_ij_meta(
    ij_meta: Dict[str, Any], new_channel_name_order: List[str]
) -> Dict[str, Any]:
    num_ch = len(new_channel_name_order)
    new_ij_meta = deepcopy(ij_meta)
    new_ij_meta["Labels"] = []
    new_ij_meta["Labels"] = new_channel_name_order
    new_ij_meta["channels"] = num_ch
    new_ij_meta["images"] = num_ch
    return new_ij_meta


def do_bg_subtraction(img: Image, bg: Image) -> Image:
    orig_dtype = deepcopy(img.dtype)
    orig_dtype_minmax = (np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max)
    return (img.astype(np.int32) - bg).clip(*orig_dtype_minmax).astype(orig_dtype)


def subtract_bg_from_imgs(
    img_path: Path,
    out_dir: Path,
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
    cycles_with_bg_ch: Dict[int, Dict[int, int]],
    fractions_of_bg_per_cycle: Dict[int, Dict[int, int]],
    nuc_ch_stack_id: int,
    bg_ch_stack_ids: List[int],
    new_channel_name_order: List[str],
):
    img_stack, ij_meta = read_stack_and_meta(img_path)
    orig_dtype = deepcopy(img_stack.dtype)
    bg_images: Dict[int, Dict[int, Image]] = {cyc: dict() for cyc in cycles_with_bg_ch.keys()}
    for cycle, channels in cycles_with_bg_ch.items():
        for ch_id, stack_id in channels.items():
            bg_images[cycle][ch_id] = img_stack[stack_id, :, :]
    print("Subtracting background from", img_path.name)
    processed_imgs = []
    for cycle, channels in stack_ids_per_cycle.items():
        if channels != {}:
            for ch_id, stack_id in channels.items():
                if stack_id == nuc_ch_stack_id:
                    processed_imgs.append(img_stack[stack_id, :, :])
                elif stack_id in bg_ch_stack_ids:
                    continue
                else:
                    fractions = fractions_of_bg_per_cycle[cycle]
                    bg_cycles = list(fractions.keys())
                    if len(bg_cycles) == 1:
                        bg_img = bg_images[bg_cycles[0]][ch_id]
                    else:
                        bg_imgs = []
                        for bg_cyc, frac in fractions.items():
                            bg = bg_images[bg_cyc][ch_id]
                            bg_imgs.append(bg * frac)
                        # sum background images from first and last cycle
                        bg_img = np.round(np.sum(bg_imgs, axis=0)).astype(orig_dtype)

                    processed_img = do_bg_subtraction(img_stack[stack_id, :, :], bg_img)
                    processed_imgs.append(processed_img)
    processed_stack = np.stack(processed_imgs, axis=0)
    del processed_imgs, processed_img, img_stack
    out_path = out_dir / img_path.name

    new_ij_meta = modify_initial_ij_meta(ij_meta, new_channel_name_order)

    save_stack(out_path, processed_stack, new_ij_meta)


def subtract_bg_from_imgs_parallelized(
    img_listing: List[Path],
    out_dir: Path,
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
    cycles_with_bg_ch: Dict[int, Dict[int, int]],
    fractions_of_bg_per_cycle: Dict[int, Dict[int, int]],
    nuc_ch_stack_id: int,
    bg_ch_stack_ids: List[int],
    new_channel_name_order: List[str],
):
    tasks = []
    for img_path in img_listing:
        task = dask.delayed(subtract_bg_from_imgs)(
            img_path,
            out_dir,
            stack_ids_per_cycle,
            cycles_with_bg_ch,
            fractions_of_bg_per_cycle,
            nuc_ch_stack_id,
            bg_ch_stack_ids,
            new_channel_name_order,
        )
        tasks.append(task)
    dask.compute(*tasks)


def main(data_dir: Path, pipeline_config_path: Path, cytokit_config_path: Path):
    """Input images are expected to be stack outputs of cytokit processing
    and have names R001_X001_Y001.tif
    """
    out_dir = Path("/output/background_subtraction")
    make_dir_if_not_exists(out_dir)

    dataset_info = load_dataset_info(pipeline_config_path)
    nuclei_channel = dataset_info["nuclei_channel"]
    num_cycles = dataset_info["num_cycles"]
    num_channels_per_cycle = dataset_info["num_channels"]
    channel_names = dataset_info["channel_names"]
    background_ch_name = "blank"

    expr_dir = data_dir / "extract/expressions"
    img_listing = get_img_listing(expr_dir)
    print(img_listing)

    print("Channel names\n", channel_names)
    print("Number of channels per cycle\n", num_channels_per_cycle)

    # will read channel names from imagej metadata in tifs
    channel_names_in_stack = get_channel_names_in_stack(img_listing[0])
    print("Channels names in stack\n", channel_names_in_stack)

    channels_per_cycle = organize_channels_per_cycle(
        channel_names, num_cycles, num_channels_per_cycle
    )
    print("Channels per cycle\n", channels_per_cycle)

    filtered_channels_per_cycle = filter_channels(channels_per_cycle, channel_names_in_stack)
    print("Filtered channels per cycle\n", filtered_channels_per_cycle)

    new_channel_name_order = create_new_channel_name_order(
        filtered_channels_per_cycle, background_ch_name
    )
    print("New channel name order", new_channel_name_order)

    # channel ids start from 1, stack ids start from 0
    stack_ids_per_cycle = get_stack_ids_per_cycle(
        filtered_channels_per_cycle, channel_names_in_stack
    )
    print("Stack ids per cycle\n", stack_ids_per_cycle)

    bg_channel_ids_per_cycle = get_stack_ids_of_bg_channels(
        filtered_channels_per_cycle, background_ch_name
    )

    nuc_ch_stack_id = get_nuc_ch_stack_id(
        nuclei_channel, filtered_channels_per_cycle, stack_ids_per_cycle
    )
    print("Nucleus ch stack id\n", nuc_ch_stack_id)
    bg_ch_stack_ids = get_bg_ch_stack_ids(
        background_ch_name, filtered_channels_per_cycle, stack_ids_per_cycle
    )
    print("Background channel stack ids\n", bg_ch_stack_ids)

    cycles_with_bg_ch = select_cycles_with_bg_ch(bg_channel_ids_per_cycle, num_channels_per_cycle)
    print("Cycles with background channels\n", cycles_with_bg_ch)
    if len(cycles_with_bg_ch) == 0:
        print("NOT ENOUGH BACKGROUND CHANNELS")
        print("WILL SKIP BACKGROUND SUBTRACTION")
        return
    cycle_names = list(bg_channel_ids_per_cycle.keys())

    if len(cycles_with_bg_ch) == 1:
        fractions_of_bg_per_cycle = assign_fraction_of_bg_mix_when_one_bg_cyc(
            cycles_with_bg_ch, cycle_names
        )
    else:
        fractions_of_bg_per_cycle = assign_fraction_of_bg_mix(cycles_with_bg_ch, cycle_names)

    print("Fractions of background per cycle\n", fractions_of_bg_per_cycle)
    subtract_bg_from_imgs_parallelized(
        img_listing,
        out_dir,
        stack_ids_per_cycle,
        cycles_with_bg_ch,
        fractions_of_bg_per_cycle,
        nuc_ch_stack_id,
        bg_ch_stack_ids,
        new_channel_name_order,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with Cytokit output")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )
    parser.add_argument("--cytokit_config_path", type=Path, help="path to experiment.yaml file")
    args = parser.parse_args()
    main(args.data_dir, args.pipeline_config_path, args.cytokit_config_path)
