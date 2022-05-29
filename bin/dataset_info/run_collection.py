import argparse
from pathlib import Path

import collect_dataset_info_old

import collect_dataset_info


def find_raw_data_dir(dataset_dir: Path):
    NONRAW_DIRECTORY_NAME_PIECES = [
        "processed",
        "drv",
        "metadata",
        "extras",
        "Overview",
    ]
    raw_data_dir_possibilities = []

    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if not any(piece in child.name for piece in NONRAW_DIRECTORY_NAME_PIECES):
            raw_data_dir_possibilities.append(child)

    if len(raw_data_dir_possibilities) > 1:
        message_pieces = ["Found multiple raw data directory possibilities:"]
        message_pieces.extend(f"\t{path}" for path in raw_data_dir_possibilities)
        raise ValueError("\n".join(message_pieces))
    raw_data_dir = raw_data_dir_possibilities[0]
    return raw_data_dir


def check_new_meta_present(raw_data_dir: Path):
    if Path(raw_data_dir / "dataset.json").exists():
        print("Found new metadata")
        return True
    else:
        print("Did not found new metadata. Will try to use old metadata")
        return False


def main(path_to_dataset: Path):
    raw_data_dir = find_raw_data_dir(path_to_dataset)
    is_new_meta_present = check_new_meta_present(raw_data_dir)
    if is_new_meta_present:
        collect_dataset_info.main(path_to_dataset)
    else:
        collect_dataset_info_old.main(path_to_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect information required to perform analysis of a CODEX dataset."
    )
    parser.add_argument(
        "--path_to_dataset",
        help="Path to directory containing raw data subdirectory (with with cycle and region numbers).",
        type=Path,
    )
    args = parser.parse_args()
    main(args.path_to_dataset)
