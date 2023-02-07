from os import scandir
from typing import Iterable
from pathlib import Path
from json import load
from metrics_as_scores.distribution.distribution import LocalDataset

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
            try:
                manifest = datasets_dir.joinpath(f'./{d.name}/manifest.json')
                if manifest.exists():
                    with open(file=str(manifest), mode='r', encoding='utf-8') as fp:
                        manifest: LocalDataset = load(fp=fp)
                        yield manifest
            except:
                pass
