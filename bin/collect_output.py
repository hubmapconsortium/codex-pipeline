import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

sys.path.append("/opt/")
import dask

from pipeline_utils.pipeline_config_reader import load_dataset_info


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_segm_listing(segm_dir: Path):
    img_list = []
    for reg_dir in list(segm_dir.glob("region_???")):
        for img_path in list(reg_dir.glob("reg???_mask.ome.tiff")):
            img_list.append(img_path)
    return img_list


def collect_segm_masks(segmentation_masks: Path, out_dir: Path):
    print("Collecting segmentation masks")
    segm_mask_img_list = get_segm_listing(segmentation_masks)
    if segm_mask_img_list == []:
        msg = "No segmentation masks found in", str(segmentation_masks)
        raise ValueError(msg)
    for img_path in segm_mask_img_list:
        shutil.copy(img_path, out_dir / img_path.name)


def get_expr_listing(expr_dir: Path):
    img_list = []
    for img_path in list(expr_dir.glob("reg???_expr.ome.tiff")):
        img_list.append(img_path)
    return img_list


def collect_expr(expression_channels: Path, out_dir: Path):
    print("Collection expression images")
    expr_img_list = get_expr_listing(expression_channels)
    if expr_img_list == []:
        msg = "No expression images found in", str(expression_channels)
        raise ValueError(msg)
    for img_path in expr_img_list:
        shutil.copy(img_path, out_dir / img_path.name)


def main(pipeline_config_path: Path, expression_channels: Path, segmentation_masks: Path):
    pipeline_config = load_dataset_info(pipeline_config_path)

    num_concurrent_tasks = pipeline_config["num_concurrent_tasks"]

    out_dir = Path("/output/pipeline_output")
    mask_out_dir = out_dir / "mask"
    expr_out_dir = out_dir / "expr"
    make_dir_if_not_exists(mask_out_dir)
    make_dir_if_not_exists(expr_out_dir)

    dask.config.set({"num_workers": num_concurrent_tasks, "scheduler": "processes"})
    print("\nOrganizing pipeline output")
    tasks = [
        dask.delayed(collect_segm_masks)(segmentation_masks, mask_out_dir),
        dask.delayed(collect_expr)(expression_channels, expr_out_dir),
    ]
    dask.compute(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config", type=Path, help="path to region map file YAML")
    parser.add_argument("--expression_channels", type=Path, help="path to directory with images")
    parser.add_argument(
        "--segmentation_masks", type=Path, help="path to directory with segmentation masks"
    )
    args = parser.parse_args()

    main(args.pipeline_config, args.expression_channels, args.segmentation_masks)
