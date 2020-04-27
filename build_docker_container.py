#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, run
import sys
from typing import List, Set, Tuple

class RefusalToBuildException(Exception):
    pass

ERROR_COLOR = '\033[01;31m'
NO_COLOR = '\033[00m'

# Would like to include timezone offset, but not worth the
# complexity of including pytz/etc.
TIMESTAMP_FORMAT = '%Y%m%d-%H%M%S%z'

DOCKER = 'docker'
DOCKER_BUILD_COMMAND_TEMPLATE: List[str] = [
    DOCKER,
    'build',
    '-q',
    '-t',
    '{label}',
    '-f',
    '{dockerfile_path}',
    '.',
]
DOCKER_TAG_COMMAND_TEMPLATE: List[str] = [
    DOCKER,
    'tag',
    '{image_id}',
    '{tag_name}',
]
DOCKER_PUSH_COMMAND_TEMPLATE: List[str] = [
    DOCKER,
    'push',
    '{image_id}',
]

GIT_SUBMODULE_STATUS_COMMAND: List[str] = [
    'git',
    'submodule',
    'status',
]

# List of (label, filename) tuples
IMAGES: List[Tuple[str, Path]] = [
    ('hubmap/codex-scripts', Path('Dockerfile')),
    ('hubmap/cytokit', Path('cytokit-docker/Dockerfile')),
]

def print_run(command: List[str], pretend: bool, return_stdout: bool=False, **kwargs):
    if 'cwd' in kwargs:
        directory_piece = f' in directory "{kwargs["cwd"]}"'
    else:
        directory_piece = ''
    print('Running "{}"{}'.format(' '.join(command), directory_piece))
    if pretend:
        return '<pretend>'
    else:
        kwargs = kwargs.copy()
        if return_stdout:
            kwargs['stdout'] = PIPE
        proc = run(command, check=True, **kwargs)
        if return_stdout:
            return proc.stdout.strip().decode('utf-8')

def check_submodules(directory: Path, ignore_missing_submodules: bool):
    submodule_status_output = run(
        GIT_SUBMODULE_STATUS_COMMAND,
        stdout=PIPE,
        cwd=directory,
    ).stdout.decode('utf-8').splitlines()

    # name, commit
    uninitialized_submodules: Set[Tuple[str, str]] = set()

    for line in submodule_status_output:
        status_code, pieces = line[0], line[1:].split()
        if status_code == '-':
            uninitialized_submodules.add((pieces[1], pieces[0]))

    if uninitialized_submodules:
        message_pieces = ['Found uninitialized submodules:']
        for name, commit in sorted(uninitialized_submodules):
            message_pieces.append(f'\t{name} (at commit {commit})')
        message_pieces.append("Maybe you need to run")
        message_pieces.append("\tgit submodule update --init")
        message_pieces.append("(Override with '--ignore-missing-submodules' if you're really sure.)")

        if not ignore_missing_submodules:
            raise RefusalToBuildException('\n'.join(message_pieces))

def main(tag_timestamp: bool, push: bool, ignore_missing_submodules: bool, pretend: bool):
    directory_of_this_script = Path(__file__).parent
    check_submodules(directory_of_this_script, ignore_missing_submodules)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    images_to_push = []
    for label_base, filename in IMAGES:
        label = f'{label_base}:latest'
        base_dir = filename.parent

        docker_build_command = [
            piece.format(
                label=label,
                dockerfile_path=filename.name,
            )
            for piece in DOCKER_BUILD_COMMAND_TEMPLATE
        ]
        image_id = print_run(docker_build_command, pretend, return_stdout=True, cwd=base_dir)
        images_to_push.append(label)
        print('Tagged image', image_id, 'as', label)

        if tag_timestamp:
            timestamp_tag_name = f'{label_base}:{timestamp}'
            docker_tag_latest_command = [
                piece.format(
                    image_id=image_id,
                    tag_name=timestamp_tag_name,
                )
                for piece in DOCKER_TAG_COMMAND_TEMPLATE
            ]
            print_run(docker_tag_latest_command, pretend)
            print('Tagged image', image_id, 'as', timestamp_tag_name)
            images_to_push.append(timestamp_tag_name)

    if push:
        for image_id in images_to_push:
            docker_push_command = [
                piece.format(
                    image_id=image_id,
                )
                for piece in DOCKER_PUSH_COMMAND_TEMPLATE
            ]
            print_run(docker_push_command, pretend)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--tag-timestamp', action='store_true')
    p.add_argument('--push', action='store_true')
    p.add_argument('--ignore-missing-submodules', action='store_true')
    p.add_argument('--pretend', action='store_true')
    args = p.parse_args()

    try:
        main(args.tag_timestamp, args.push, args.ignore_missing_submodules, args.pretend)
    except RefusalToBuildException as e:
        print(ERROR_COLOR + 'Refusing to build Docker containers, for reason:' + NO_COLOR)
        sys.exit(e.args[0])
