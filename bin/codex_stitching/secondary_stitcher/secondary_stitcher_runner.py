import argparse
import json
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

import secondary_stitcher

Report = Dict[str, Dict[str, Any]]


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def read_pipeline_config(path_to_config: Path) -> dict:
    with open(path_to_config, "r") as s:
        config = json.load(s)
    return config


def write_pipeline_config(out_path: Path, config):
    with open(out_path, "w") as s:
        json.dump(config, s, sort_keys=False, indent=4)


def run_stitcher(
    img_dir: Path,
    out_dir: Path,
    img_name_template: str,
    overlap: int,
    padding: dict,
    is_mask: bool,
    nucleus_channel: str,
    cell_channel: str,
) -> Report:
    padding_str = ",".join((str(i) for i in list(padding.values())))
    report = secondary_stitcher.main(
        img_dir,
        out_dir,
        img_name_template,
        overlap,
        padding_str,
        is_mask,
        nucleus_channel,
        cell_channel,
    )
    return report


def merge_reports(mask_report: Report, expr_report: Report) -> Report:
    total_report = dict()
    for region in mask_report:
        total_report[region] = {**mask_report[region], **expr_report[region]}
    return total_report


def main(pipeline_config_path: Path, ometiff_dir: Path):
    pipeline_config = read_pipeline_config(pipeline_config_path)
    slicer_meta = pipeline_config["slicer"]
    nucleus_channel = pipeline_config.get("nuclei_channel", "None")
    cell_channel = pipeline_config.get("membrane_channel", "None")

    path_to_mask_tiles = Path(ometiff_dir).joinpath("cytometry/tile/ome-tiff")
    path_to_image_tiles = Path(ometiff_dir).joinpath("extract/expressions/ome-tiff")

    overlap = slicer_meta["overlap"]
    padding = slicer_meta["padding"]

    mask_out_dir = Path("output/pipeline_output/mask")
    expr_out_dir = Path("output/pipeline_output/expr")
    final_pipeline_config_path = Path("output/pipelineConfig.json")

    make_dir_if_not_exists(mask_out_dir)
    make_dir_if_not_exists(expr_out_dir)

    mask_out_name_template = "reg{r:03d}_mask.ome.tiff"
    expr_out_name_template = "reg{r:03d}_expr.ome.tiff"

    mask_report = run_stitcher(
        path_to_mask_tiles,
        mask_out_dir,
        mask_out_name_template,
        overlap,
        padding,
        True,
        nucleus_channel,
        cell_channel,
    )

    expr_report = run_stitcher(
        path_to_image_tiles,
        expr_out_dir,
        expr_out_name_template,
        overlap,
        padding,
        False,
        nucleus_channel,
        cell_channel,
    )

    total_report = merge_reports(mask_report, expr_report)

    final_pipeline_config = pipeline_config
    final_pipeline_config.update({"report": total_report})
    print("\nfinal_pipeline_config")
    pprint(final_pipeline_config, sort_dicts=False)
    write_pipeline_config(final_pipeline_config_path, final_pipeline_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config_path", type=Path, help="path to pipeline config")
    parser.add_argument(
        "--ometiff_dir", type=Path, help="dir with segmentation mask tiles and codex image tiles"
    )

    args = parser.parse_args()
    main(args.pipeline_config_path, args.ometiff_dir)
