from os import scandir
from typing import Iterable
from pathlib import Path
from json import load, loads
from urllib.request import urlopen
from metrics_as_scores.distribution.distribution import LocalDataset, KnownDataset

this_dir = Path(__file__).resolve().parent
datasets_dir = this_dir.parent.parent.parent.joinpath('./datasets')




def isint(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False

def isnumeric(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False


def get_local_datasets() -> Iterable[LocalDataset]:
    for d in scandir(path=str(datasets_dir.resolve())):
        if d.is_dir():
            if d.name == '_default':
                continue
            try:
                manifest = datasets_dir.joinpath(f'./{d.name}/manifest.json')
                if manifest.exists():
                    with open(file=str(manifest), mode='r', encoding='utf-8') as fp:
                        manifest: LocalDataset = load(fp=fp)
                        yield manifest
            except:
                pass


KNOWN_DATASETS_FILE = 'https://raw.githubusercontent.com/MrShoenel/metrics-as-scores/master/datasets/known-datasets.json'

def get_known_datasets(use_local_file: bool=False) -> list[KnownDataset]:
    if use_local_file:
        with open(file=str(datasets_dir.joinpath(f'./known-datasets.json')), mode='r', encoding='utf-8') as fp:
            return load(fp=fp)
    return loads(urlopen(url=KNOWN_DATASETS_FILE).read().decode('utf-8'))


def format_file_size(num_bytes: int) -> str:
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(num_bytes)

    i = 0
    while size > 1e3:
        size /= 1e3
        i += 1
    
    return f'{round(size, 2)} {suffixes[i]}'
