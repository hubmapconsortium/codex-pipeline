#!/usr/bin/env python3.5
# Note: this version ^^^ is what's available in the Cytokit image
# and our extension. No f-strings or PEP 519.

from argparse import ArgumentParser
from os import environ
from os.path import split as osps
from pathlib import Path
from subprocess import check_call

import yaml

# TODO ↓↓↓ unify this script with setting up the data directory
#  instead of calling this script as a separate executable
SETUP_DATA_DIR_COMMAND = [
    "/opt/setup_data_directory.py",
    "{new_tiles",
]
CYTOKIT_COMMAND = [
    "cytokit",
    "{command}",
    "run_all",
    "--config-path={yaml_config}",
    "--new_tiles={new_tiles}",
    "--output-dir=output",
]

CYTOKIT_PROCESSOR_OUTPUT_DIRS = frozenset({"cytometry", "processor"})


def symlink_images(data_dir: Path):
    # TODO: unify, don't call another command-line script
    command = [piece.format(data_dir=data_dir) for piece in SETUP_DATA_DIR_COMMAND]
    print("Running:", " ".join(command))
    check_call(command)


def find_cytokit_processor_output_r(directory: Path):
    """
    BIG HACK for step-by-step CWL usage -- walk parent directories until
    we find one containing 'cytometry' and 'processor'
    """
    child_names = {c.name for c in directory.iterdir()}
    if CYTOKIT_PROCESSOR_OUTPUT_DIRS <= child_names:
        return directory
    else:
        abs_dir = directory.absolute()
        parent = abs_dir.parent
        if parent == abs_dir:
            # At the root. No data found.
            return
        else:
            return find_cytokit_processor_output_r(parent)


def find_cytokit_processor_output(directory: Path) -> Path:
    data_dir = find_cytokit_processor_output_r(directory)
    if data_dir is None:
        message = "No `cytokit processor` output found in {} or any parent directories"
        raise ValueError(message.format(directory))
    else:
        return data_dir


def find_or_prep_data_directory(cytokit_command: str, data_dir: Path, pipeline_config: Path):
    """
    :return: 2-tuple: pathlib.Path to data directory, either original or
     newly-created with symlinks
    """
    # Read directory name from pipeline config
    # Python 3.6 would be much nicer ,but the Cytokit image is built from
    # Ubuntu 16.04, which comes with 3.5
    with pipeline_config.open() as f:
        config = yaml.safe_load(f)
    dir_name = osps(config["raw_data_location"])[1]

    data_subdir = data_dir / dir_name

    if cytokit_command == "processor":
        symlink_images(data_subdir)
        return Path("symlinks")
    elif cytokit_command == "operator":
        # Need to find the output from 'cytokit processor'
        processor_dir = find_cytokit_processor_output(data_dir)
        output_path = Path("output")
        output_path.mkdir()
        for child in processor_dir.iterdir():
            link = output_path / child.name
            print("Symlinking", child, "to", link)
            link.symlink_to(child)
        return output_path
    else:
        raise ValueError('Unsupported Cytokit command: "{}"'.format(cytokit_command))


def run_cytokit(cytokit_command: str, data_directory: Path, yaml_config: Path):
    command = [
        piece.format(
            command=cytokit_command,
            data_dir=data_directory,
            yaml_config=yaml_config,
        )
        for piece in CYTOKIT_COMMAND
    ]
    print("Running:", " ".join(command))
    env = environ.copy()
    env["PYTHONPATH"] = "/lab/repos/cytokit/python/pipeline"
    check_call(command, env=env)

    print("Cytokit completed successfully")
    # I feel really bad about this, but not bad enough not to do it
    if cytokit_command == "operator":
        output_dir = Path("output")
        for dirname in CYTOKIT_PROCESSOR_OUTPUT_DIRS:
            dir_to_delete = output_dir / dirname
            print("Deleting", dir_to_delete)
            dir_to_delete.unlink()


def main(cytokit_command: str, data_dir: Path, pipeline_config: Path, yaml_config: Path):
    data_dir = find_or_prep_data_directory(cytokit_command, data_dir, pipeline_config)
    run_cytokit(cytokit_command, data_dir, yaml_config)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("cytokit_command")
    p.add_argument("new_tiles", type=Path)
    p.add_argument("pipeline_config", type=Path)
    p.add_argument("yaml_config", type=Path)
    args = p.parse_args()

    main(args.cytokit_command, args.new_tiles, args.pipeline_config, args.yaml_config)
