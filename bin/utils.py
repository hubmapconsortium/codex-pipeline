from pathlib import Path
from pprint import pprint

def print_directory_tree(directory: Path):
    pprint(sorted(directory.glob('**/*')))
