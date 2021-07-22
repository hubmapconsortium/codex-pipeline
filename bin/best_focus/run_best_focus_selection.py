import argparse
import os
import sys
from pathlib import Path
from typing import List

sys.path.append("/opt/")
from best_z_paths import find_best_z_paths_and_dirs
from file_manipulation import process_z_planes_and_save_to_out_dirs

from pipeline_utils.pipeline_config_reader import load_dataset_info


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_img_dirs(dataset_dir: Path) -> List[Path]:
    dataset_dir = dataset_dir.absolute()
    img_dir_names = next(os.walk(dataset_dir))[1]
    img_dir_paths = [dataset_dir.joinpath(dir_name).absolute() for dir_name in img_dir_names]
    return img_dir_paths


def main(data_dir: Path, pipeline_config_path: Path):
    best_focus_dir = Path("/output/best_focus")
    make_dir_if_not_exists(best_focus_dir)
    dataset_info = load_dataset_info(pipeline_config_path)
    img_dirs = get_img_dirs(data_dir)
    best_z_channel_dirs, best_z_plane_paths = find_best_z_paths_and_dirs(
        dataset_info, img_dirs, best_focus_dir
    )
    process_z_planes_and_save_to_out_dirs(best_z_channel_dirs, best_z_plane_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="path to directory with dataset directory")
    parser.add_argument(
        "--pipeline_config_path", type=Path, help="path to pipelineConfig.json file"
    )
    args = parser.parse_args()
    main(args.data_dir, args.pipeline_config_path)
