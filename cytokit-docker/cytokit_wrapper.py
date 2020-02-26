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
    '/opt/setup_data_directory.py',
    '{data_dir}',
]
CYTOKIT_COMMAND = [
    'cytokit',
    '{command}',
    'run_all',
    '--config-path={yaml_config}',
    '--data-dir={data_dir}',
    '--output-dir=output',
]


def symlink_images(data_dir: Path, pipeline_config: Path):
    # Read directory name from pipeline config
    # Python 3.6 would be much nicer ,but the Cytokit image is built from
    # Ubuntu 16.04, which comes with 3.5
    with pipeline_config.open() as f:
        config = yaml.safe_load(f)
    dir_name = osps(config['raw_data_location'])[1]

    data_subdir = data_dir / dir_name
    # TODO: unify, don't call another command-line script
    command = [
        piece.format(data_dir=data_subdir)
        for piece in SETUP_DATA_DIR_COMMAND
    ]
    print('Running:', ' '.join(command))
    check_call(command)


def prep_data_directory(cytokit_command: str, data_dir: Path, pipeline_config: Path):
    """
    :return: 2-tuple: pathlib.Path to data directory, either original or
     newly-created with symlinks
    """
    if cytokit_command == 'processor':
        symlink_images(data_dir, pipeline_config)
        return Path('symlinks')
    elif cytokit_command == 'operator':
        return data_dir
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
    print('Running:', ' '.join(command))
    env = environ.copy()
    env['PYTHONPATH'] = '/lab/repos/cytokit/python/pipeline'
    check_call(command, env=env)


def main(cytokit_command: str, data_dir: Path, pipeline_config: Path, yaml_config: Path):
    data_dir = prep_data_directory(cytokit_command, data_dir, pipeline_config)
    run_cytokit(cytokit_command, data_dir, yaml_config)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('cytokit_command')
    p.add_argument('data_dir', type=Path)
    p.add_argument('pipeline_config', type=Path)
    p.add_argument('yaml_config', type=Path)
    args = p.parse_args()

    main(args.cytokit_command, args.data_dir, args.pipeline_config, args.yaml_config)
