import argparse
import json
import re
import sys
from copy import deepcopy
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dask
import numpy as np
import tifffile as tif

# from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_otsu

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


def get_channel_names_in_stack(img_path: Path) -> List[str]:
    with tif.TiffFile(str(img_path.absolute())) as TF:
        ij_meta = TF.imagej_metadata
    channel_names = [ch_name.lstrip("proc_").lower() for ch_name in ij_meta["Labels"]]
    return channel_names


def get_stack_ids_per_cycle(
    channels_per_cycle: Dict[int, Dict[int, str]],
    channel_names_in_stack: List[str],
    bg_ch_name: str,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]]]:
    stack_ids_per_cycle = {cyc: dict() for cyc in channels_per_cycle.keys()}
    lookup_ch_name_list = [ch.lower() for ch in channel_names_in_stack]

    bg_channel_ids_per_cycle = {cyc: dict() for cyc in channels_per_cycle.keys()}
    bg_ch_pattern = re.compile(r"^" + bg_ch_name, re.IGNORECASE)

    for cycle, channels in channels_per_cycle.items():
        for ch_id, ch_name in channels.items():
            ch_name_low = ch_name.lower()
            if ch_name_low in lookup_ch_name_list:
                stack_id = lookup_ch_name_list.index(ch_name_low)
                lookup_ch_name_list[stack_id] = None
                stack_ids_per_cycle[cycle][ch_id] = stack_id
                if bg_ch_pattern.match(ch_name):
                    bg_channel_ids_per_cycle[cycle][ch_id] = stack_id
    return stack_ids_per_cycle, bg_channel_ids_per_cycle


def select_cycles_with_bg_ch(
    bg_channel_ids_per_cycle: Dict[int, Dict[int, int]], num_channels_per_cycle: int
) -> Dict[int, Dict[int, int]]:
    required_num_of_bg_channels = num_channels_per_cycle - 1
    selected_bg_channels_per_cyc = dict()
    for cycle, channels in bg_channel_ids_per_cycle.items():
        if len(channels.keys()) >= required_num_of_bg_channels:
            selected_bg_channels_per_cyc[cycle] = channels
    return selected_bg_channels_per_cyc


def get_ch_stack_ids(
    target_ch_name: str,
    channel_names_in_stack: List[str],
    channels_per_cycle: Dict[int, Dict[int, str]],
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
) -> List[int]:
    pat = re.compile(r"^" + target_ch_name, re.IGNORECASE)
    target_ch_stack_ids = []
    channel_names_in_stack = set(channel_names_in_stack)
    for cycle in channels_per_cycle:
        for ch_id, ch_name in channels_per_cycle[cycle].items():
            if pat.match(ch_name) and ch_name.lower() in channel_names_in_stack:
                target_ch_stack_id = stack_ids_per_cycle[cycle][ch_id]
                target_ch_stack_ids.append(target_ch_stack_id)
    return target_ch_stack_ids


def assign_fraction_of_bg_mix_when_one_bg_cyc(
    expr_cycles: Dict[int, Dict[int, int]], bg_cycles: Dict[int, Dict[int, int]]
) -> Dict[int, Dict[int, int]]:
    # {3: {1: 1},}
    expr_cycles = sorted(list(expr_cycles.keys()))
    bg_cycles = sorted(list(bg_cycles.keys()))

    first_bg_cycle = bg_cycles[0]
    last_expr_cycle = expr_cycles[-1]
    first_bg_cycle_id = expr_cycles.index(first_bg_cycle)
    last_expr_cycle_id = expr_cycles.index(last_expr_cycle)

    fractions_per_cycle = dict()

    cycle_subset = expr_cycles[first_bg_cycle_id + 1 : last_expr_cycle_id + 1]
    fractions_per_cycle[first_bg_cycle] = {first_bg_cycle: {}}
    for i, cycle in enumerate(cycle_subset):
        fractions_per_cycle[cycle] = {first_bg_cycle: 1}
    fractions_per_cycle_sorted = {
        k: v for k, v in sorted(fractions_per_cycle.items(), key=lambda item: item[0])
    }
    return fractions_per_cycle_sorted


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}


def get_medq(img: Image, q: float = 0.8) -> float:
    # returns median of some quantile
    thr = np.quantile(img, q)
    medq = np.median(img[img <= thr])
    return medq


def get_otsu_bg_val(img: Image) -> float:
    thresh = threshold_otsu(img)
    below_th = img <= thresh
    below_med = np.nanmedian(img[below_th])
    return below_med


def lin_fit(y, deg=1) -> np.ndarray:
    ids = np.arange(1, len(y) + 1)
    coef = np.polyfit(ids, y, deg)
    poly1d_fn = np.poly1d(coef)
    slope, intercept = coef
    fit = poly1d_fn(ids)
    return fit, slope, intercept


def calc_background_median(
    img_path: Path, stack_ids_per_cycle: Dict[int, Dict[int, int]]
) -> Dict[int, Dict[int, float]]:
    stack = tif.imread(img_path)
    med_per_ch_cyc = dict()

    for cyc in stack_ids_per_cycle:
        for ch, stack_id in stack_ids_per_cycle[cyc].items():
            this_ch_img = stack[stack_id, :, :]
            bg_val = get_otsu_bg_val(this_ch_img)
            if ch in med_per_ch_cyc:
                med_per_ch_cyc[ch][cyc] = bg_val
            else:
                med_per_ch_cyc[ch] = {cyc: bg_val}
    return med_per_ch_cyc


def filter_bg_fractions(
    bg_fractions: Dict[int, Dict[int, float]]
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float], Dict[int, float]]:
    # most of the operations in this function is just reordering of dictionaries
    fr_per_ch = dict()
    cyc_ids_per_ch = dict()
    for cyc in bg_fractions:
        for ch, fr in bg_fractions[cyc].items():
            if ch in fr_per_ch:
                fr_per_ch[ch].append(fr)
                cyc_ids_per_ch[ch].append(cyc)
            else:
                fr_per_ch[ch] = [fr]
                cyc_ids_per_ch[ch] = [cyc]

    # linear fitting
    slope_per_ch = dict()
    intercept_per_ch = dict()
    fr_per_ch_filtered = dict()
    for ch in fr_per_ch:
        fr_list = fr_per_ch[ch]
        filtered_fr, slope, intercept = lin_fit(fr_list)
        fr_per_ch_filtered[ch] = filtered_fr
        slope_per_ch[ch] = slope
        intercept_per_ch[ch] = intercept

    bg_fractions_per_ch_cor = dict()
    for ch in fr_per_ch_filtered:
        this_ch_cyc_fr = {k: v for k, v in zip(cyc_ids_per_ch[ch], fr_per_ch_filtered[ch])}
        bg_fractions_per_ch_cor[ch] = this_ch_cyc_fr

    bg_fractions_filtered = dict()
    for ch in bg_fractions_per_ch_cor:
        for cyc, fr in bg_fractions_per_ch_cor[ch].items():
            if cyc in bg_fractions_filtered:
                bg_fractions_filtered[cyc].update({ch: fr})
            else:
                bg_fractions_filtered[cyc] = {ch: fr}
    return bg_fractions_filtered, slope_per_ch, intercept_per_ch


def estimate_background_fraction_when_one_bg_cycle(
    img_listing: List[Path], stack_ids_per_cycle: Dict[int, Dict[int, int]]
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float], Dict[int, float]]:
    tasks = []
    for img_path in img_listing:
        task = dask.delayed(calc_background_median)(img_path, stack_ids_per_cycle)
        tasks.append(task)
    meds_per_img_ch_cyc = dask.compute(*tasks)

    med_per_ch_cyc_across_imgs = dict()
    for med_per_ch_cyc in meds_per_img_ch_cyc:
        for ch in med_per_ch_cyc:

            if ch in med_per_ch_cyc_across_imgs:
                pass
            else:
                med_per_ch_cyc_across_imgs[ch] = dict()

            for cyc, med in med_per_ch_cyc[ch].items():
                if cyc in med_per_ch_cyc_across_imgs[ch]:
                    med_per_ch_cyc_across_imgs[ch][cyc].append(med)
                else:
                    med_per_ch_cyc_across_imgs[ch].update({cyc: [med]})

    med_per_ch_cyc_final = dict()
    for ch in med_per_ch_cyc_across_imgs:
        med_per_ch_cyc_final[ch] = dict()
        for cyc, meds in med_per_ch_cyc_across_imgs[ch].items():

            med_med = np.median(meds)

            if ch in med_per_ch_cyc_final:
                med_per_ch_cyc_final[ch][cyc] = med_med
            else:
                med_per_ch_cyc_final[ch].update({cyc: med_med})

    bg_fractions = dict()
    for ch in med_per_ch_cyc_final:
        first_cyc = min(list(med_per_ch_cyc_final[ch].keys()))
        for cyc, this_cyc_med in med_per_ch_cyc_final[ch].items():
            if cyc in bg_fractions:
                pass
            else:
                bg_fractions[cyc] = dict()
            first_cyc_med = med_per_ch_cyc_final[ch][first_cyc]
            bg_fraction = round(first_cyc_med / this_cyc_med, 3)

            if cyc in bg_fractions:
                bg_fractions[cyc][ch] = bg_fraction
            else:
                bg_fractions[cyc].update({ch: bg_fraction})
    bg_fractions = sort_dict(bg_fractions)

    bg_fractions_filtered, slope_per_ch, intercept_per_ch = filter_bg_fractions(bg_fractions)
    return bg_fractions_filtered, slope_per_ch, intercept_per_ch


def assign_fraction_of_bg_mix(
    stack_ids_per_cycle: Dict[int, Dict[int, int]], cycles_with_bg_ch: Dict[int, Dict[int, int]]
) -> Dict[int, Dict[int, int]]:
    """
    The interpolation used here assumes that the background (due to autofluourescence) varies
    linearly with time, and that each cycle takes the same amount of time.  The coordinates for
    the interpolation place the zero point of time at the center of the first cycle.
    """
    expr_cycles = sorted(stack_ids_per_cycle)
    bg_cycles = sorted(cycles_with_bg_ch)

    first_bg_cycle = bg_cycles[0]
    last_bg_cycle = bg_cycles[-1]
    assert len(bg_cycles) <= 2, "More than 2 background cycles are not supported"
    first_bg_cycle_id = expr_cycles.index(first_bg_cycle)
    last_bg_cycle_id = expr_cycles.index(last_bg_cycle)
    first_cycle_id = expr_cycles.index(expr_cycles[0])
    last_cycle_id = expr_cycles.index(expr_cycles[-1])
    if not all(a + 1 == b for a, b in zip(range(last_cycle_id + 1), expr_cycles)):
        raise AssertionError("Not all cycles appear in the stack?")
    slope, intercept = np.polyfit(
        [float(first_bg_cycle_id), float(last_bg_cycle_id)],
        [0.0, 1.0],
        deg=1,
    )

    fractions_per_cycle = {}
    for idx in range(last_cycle_id + 1):
        frac = intercept + float(idx) * slope
        fractions_per_cycle[idx + 1] = {
            first_bg_cycle: 1.0 - frac,
            last_bg_cycle: frac,
        }
    fractions_per_cycle[first_bg_cycle] = {first_bg_cycle: {}, last_bg_cycle: {}}
    fractions_per_cycle[last_bg_cycle] = {first_bg_cycle: {}, last_bg_cycle: {}}

    # Does this bit actually accomplish anything?  It is a dict, after all.
    fractions_per_cycle_sorted = {
        k: v for k, v in sorted(fractions_per_cycle.items(), key=lambda item: item[0])
    }

    slope_per_ch = dict()
    intercept_per_ch = dict()
    for ch in stack_ids_per_cycle[expr_cycles[0]]:
        slope_per_ch[ch] = slope
        intercept_per_ch[ch] = intercept
    return fractions_per_cycle_sorted, slope_per_ch, intercept_per_ch


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
    return np.clip(img.astype(np.int32) - bg, *orig_dtype_minmax).astype(orig_dtype)


def sum_bounded(arr_list: List[Image], dtype) -> Image:
    dtype_minmax = (np.iinfo(dtype).min, np.iinfo(dtype).max)
    img_sum = np.clip(
        np.round(np.sum(arr_list, axis=0, dtype=np.float32), 0), *dtype_minmax
    ).astype(dtype)
    return img_sum


def subtract_bg_from_imgs(
    img_path: Path,
    out_dir: Path,
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
    cycles_with_bg_ch: Dict[int, Dict[int, int]],
    fractions_of_bg_per_cycle: Dict[int, Dict[int, int]],
    nuc_ch_stack_id: int,
    bg_ch_stack_ids: List[int],
    new_ch_names: List[str],
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
                    fraction_map = fractions_of_bg_per_cycle[cycle]
                    # fraction_map has different structure
                    # depending on the number of bg cycles.
                    # If there is 1 bg cycle, then  fraction_map = {cyc: {ch: frac}}
                    # If there are 2 bg cycles, then fraction_map = {cyc: {bg_cyc: frac}}

                    if len(cycles_with_bg_ch) == 1:
                        bg_cycles = sorted(list(cycles_with_bg_ch.keys()))
                        first_bg_cyc = bg_cycles[0]
                        bg_frac = fraction_map[ch_id]
                        inv_frac = round(1 / bg_frac, 3)
                        bg_img = (
                            bg_images[first_bg_cyc][ch_id].astype(np.float32) * inv_frac
                        ).astype(orig_dtype)
                    else:
                        bg_imgs = []
                        for bg_cyc, frac in fraction_map.items():
                            bg = bg_images[bg_cyc][ch_id]
                            fbg = bg.astype(np.float32) * frac
                            bg_imgs.append(fbg)
                        # sum background images from first and last cycle
                        bg_img = sum_bounded(bg_imgs, orig_dtype)
                    processed_img = do_bg_subtraction(img_stack[stack_id, :, :], bg_img)
                    processed_imgs.append(processed_img)
    processed_stack = np.stack(processed_imgs, axis=0)
    del processed_imgs, processed_img, img_stack
    out_path = out_dir / img_path.name

    new_ij_meta = modify_initial_ij_meta(ij_meta, new_ch_names)

    save_stack(out_path, processed_stack, new_ij_meta)


def subtract_bg_from_imgs_parallelized(
    img_listing: List[Path],
    out_dir: Path,
    stack_ids_per_cycle: Dict[int, Dict[int, int]],
    cycles_with_bg_ch: Dict[int, Dict[int, int]],
    fractions_of_bg_per_cycle: Dict[int, Dict[int, int]],
    nuc_ch_stack_id: int,
    bg_ch_stack_ids: List[int],
    new_ch_names: List[str],
):
    tasks = []
    for img_path in img_listing:
        print(img_path)
        task = dask.delayed(subtract_bg_from_imgs)(
            img_path,
            out_dir,
            stack_ids_per_cycle,
            cycles_with_bg_ch,
            fractions_of_bg_per_cycle,
            nuc_ch_stack_id,
            bg_ch_stack_ids,
            new_ch_names,
        )
        tasks.append(task)
    dask.compute(*tasks)


def create_new_channel_name_order(
    channels_per_cycle: Dict[int, Dict[int, str]],
    channels_in_stack: List[str],
    background_channel: str,
) -> List[str]:
    channel_names = []
    bg_ch_pattern = re.compile(r"^" + background_channel, re.IGNORECASE)
    for cycle in channels_per_cycle:
        for ch_id, ch_name in channels_per_cycle[cycle].items():
            if bg_ch_pattern.match(ch_name):
                continue
            else:
                if any(re.match(ch_name, ch, re.IGNORECASE) for ch in channels_in_stack):
                    channel_names.append("proc_" + ch_name)
    return channel_names


def write_bg_info_to_config(
    pipeline_config_path: Path, out_dir: Path, slope_per_ch, interception_per_ch, num_bg_cyc
):
    with open(pipeline_config_path, "r") as s:
        config = json.load(s)
    config["background_info"] = {
        "num_background_cycles": num_bg_cyc,
        "slope_per_channel": slope_per_ch,
        "interception_per_channel": interception_per_ch,
    }
    with open(out_dir / "pipelineConfig.json", "w") as s:
        json.dump(config, s, sort_keys=False, indent=4)
    return


def main(
    data_dir: Path,
    pipeline_config_path: Path,
    cytokit_config_path: Path,
    out_base_dir: Path = None,
):
    """Input images are expected to be stack outputs of cytokit processing
    and have names R001_X001_Y001.tif
    """
    if out_base_dir is None:
        out_base_dir = ""  # resulting in output directly below /
    out_dir = Path(f"{out_base_dir}/output/background_subtraction")
    config_out_dir = Path(f"{out_base_dir}/output/config")
    make_dir_if_not_exists(out_dir)
    make_dir_if_not_exists(config_out_dir)

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

    # channel ids start from 1, stack ids start from 0
    stack_ids_per_cycle, bg_channel_ids_per_cycle = get_stack_ids_per_cycle(
        channels_per_cycle, channel_names_in_stack, background_ch_name
    )
    print("Stack ids per cycle\n", stack_ids_per_cycle)

    nuc_ch_stack_id = get_ch_stack_ids(
        target_ch_name=nuclei_channel,
        channel_names_in_stack=channel_names_in_stack,
        channels_per_cycle=channels_per_cycle,
        stack_ids_per_cycle=stack_ids_per_cycle,
    )
    print("Nucleus ch stack id\n", nuc_ch_stack_id)
    bg_ch_stack_ids = get_ch_stack_ids(
        target_ch_name=background_ch_name,
        channel_names_in_stack=channel_names_in_stack,
        channels_per_cycle=channels_per_cycle,
        stack_ids_per_cycle=stack_ids_per_cycle,
    )
    print("Background channel stack ids\n", bg_ch_stack_ids)

    cycles_with_bg_ch = select_cycles_with_bg_ch(bg_channel_ids_per_cycle, num_channels_per_cycle)
    print("Cycles with background channels\n", cycles_with_bg_ch)

    if len(cycles_with_bg_ch) == 0:
        print("NOT ENOUGH BACKGROUND CHANNELS")
        print("WILL SKIP BACKGROUND SUBTRACTION")
        slope_per_ch = "None"
        intercept_per_ch = "None"
        do_bg_sub = False
    elif len(cycles_with_bg_ch) == 1:
        (
            fractions_of_bg_per_cycle,
            slope_per_ch,
            intercept_per_ch,
        ) = estimate_background_fraction_when_one_bg_cycle(img_listing, stack_ids_per_cycle)
        do_bg_sub = True
    else:
        fractions_of_bg_per_cycle, slope_per_ch, intercept_per_ch = assign_fraction_of_bg_mix(
            stack_ids_per_cycle, cycles_with_bg_ch
        )
        do_bg_sub = True

    if do_bg_sub:
        print("Fractions of background per cycle\n", fractions_of_bg_per_cycle)

        new_channel_names = create_new_channel_name_order(
            channels_per_cycle, channel_names_in_stack, background_ch_name
        )
        print("New channel name order", new_channel_names)

        print("Fractions of background per cycle\n", fractions_of_bg_per_cycle)
        subtract_bg_from_imgs_parallelized(
            img_listing,
            out_dir,
            stack_ids_per_cycle,
            cycles_with_bg_ch,
            fractions_of_bg_per_cycle,
            nuc_ch_stack_id[0],
            bg_ch_stack_ids,
            new_channel_names,
        )
    write_bg_info_to_config(
        pipeline_config_path,
        config_out_dir,
        slope_per_ch,
        intercept_per_ch,
        len(cycles_with_bg_ch),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with Cytokit output")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )
    parser.add_argument("--cytokit_config_path", type=Path, help="path to experiment.yaml file")
    parser.add_argument("--out_base_dir", type=Path, help="base path for output", default=None)
    parser.add_argument(
        "--num_concurrent_tasks", type=int, help="How many worker threads", default=None
    )
    args = parser.parse_args()
    if args.num_concurrent_tasks is not None:
        dask.config.set(pool=Pool(args.num_concurrent_tasks))
    main(
        args.data_dir,
        args.pipeline_config_path,
        args.cytokit_config_path,
        out_base_dir=args.out_base_dir,
    )
