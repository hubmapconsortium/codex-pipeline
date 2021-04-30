import argparse
import json
from pathlib import Path
from pprint import pprint

import secondary_stitcher


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def read_pipeline_config(path_to_config: Path):
    with open(path_to_config, "r") as s:
        config = json.load(s)
    return config


def write_pipeline_config(out_path: Path, config):
    with open(out_path, "w") as s:
        json.dump(config, s, sort_keys=False, indent=4)


def run_stitcher(img_dir: Path, out_path: Path, overlap: int, padding: dict, is_mask: bool):
    padding_str = ",".join((str(i) for i in list(padding.values())))
    report = secondary_stitcher.main(img_dir, out_path, overlap, padding_str, is_mask)
    return report


def main(pipeline_config_path: Path, ometiff_dir: Path):

    pipeline_config = read_pipeline_config(pipeline_config_path)
    slicer_meta = pipeline_config["slicer"]

    path_to_mask_tiles = Path(ometiff_dir).joinpath("cytometry/tile/ome-tiff")
    path_to_image_tiles = Path(ometiff_dir).joinpath("extract/expressions/ome-tiff")

    overlap = slicer_meta["overlap"]
    padding = slicer_meta["padding"]

    mask_out_dir = Path("/output/stitched/mask")
    expressions_out_dir = Path("/output/stitched/expressions")
    final_pipeline_config_path = Path("/output/pipelineConfig.json")

    make_dir_if_not_exists(mask_out_dir)
    make_dir_if_not_exists(expressions_out_dir)

    stitched_mask_out_path = mask_out_dir.joinpath(Path("stitched_mask.ome.tiff"))
    stitched_expressions_out_path = expressions_out_dir.joinpath(
        Path("stitched_expressions.ome.tiff")
    )

    report = run_stitcher(
        path_to_mask_tiles, stitched_mask_out_path, overlap, padding, is_mask=True
    )

    _ = run_stitcher(
        path_to_image_tiles, stitched_expressions_out_path, overlap, padding, is_mask=False
    )

    final_pipeline_config = pipeline_config
    final_pipeline_config.update({"report": report})
    print("\nfinal_pipeline_config")
    pprint(final_pipeline_config)
    write_pipeline_config(final_pipeline_config_path, final_pipeline_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config_path", type=Path, help="path to pipeline config")
    parser.add_argument(
        "--ometiff_dir", type=Path, help="dir with segmentation mask tiles and codex image tiles"
    )

    args = parser.parse_args()
    main(args.pipeline_config_path, args.ometiff_dir)
