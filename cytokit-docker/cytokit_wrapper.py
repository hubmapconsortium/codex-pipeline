#!/usr/bin/env python3.5
# Note: this version ^^^ is what's available in the Cytokit image
# and our extension. No f-strings or PEP 519.

from argparse import ArgumentParser
from subprocess import check_call

SETUP_DATA_DIR_COMMAND = [
    '/opt/setup_data_directory.py',
    '{data_dir}',
]

CYTOKIT_COMMAND = [
    'cytokit',
    'processor',
    'run_all',
    '--config-path={yaml_config}',
    '--data-dir={data_dir}',
    '--output-dir=output',
]


def symlink_images(data_dir):
    # TODO: unify, don't call another command-line script
    command = [
        piece.format(data_dir=data_dir)
        for piece in SETUP_DATA_DIR_COMMAND
    ]
    print('Running:', ' '.join(command))
    check_call(command)


def run_cytokit(data_dir, yaml_config):
    command = [
        piece.format(
            data_dir=data_dir,
            yaml_config=CYTOKIT_COMMAND,
        )
        for piece in SETUP_DATA_DIR_COMMAND
    ]
    print('Running:', ' '.join(command))
    check_call(command)


def main(data_dir, yaml_config):
    symlink_images(data_dir)
    run_cytokit(data_dir, yaml_config)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('data_dir')
    p.add_argument('yaml_config')
    args = p.parse_args()

    main(args.data_dir, args.yaml_config)
