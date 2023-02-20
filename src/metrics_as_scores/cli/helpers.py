from os import scandir
from typing import Iterable, Union
from pathlib import Path
from json import load, loads
from urllib.request import urlopen
from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.distribution.distribution import LocalDataset, KnownDataset


KNOWN_DATASETS_FILE = 'https://raw.githubusercontent.com/MrShoenel/metrics-as-scores/master/src/metrics_as_scores/datasets/known-datasets.json'
"""
This is the URL to the curated list of available datasets to be used
with Metrics As Scores.
"""

def isint(s: str) -> bool:
    """
    Attempts to convert the string to an integer.

    s: ``str``
        The string to check.

    :rtype: ``bool``

    :returns:
        `True` if the string can be converted to an `int`; `False`, otherwise.
    """
    if not isinstance(s, str):
        return False
    
    try:
        int(s)
        return True
    except:
        return False


def isnumeric(s: Union[str, int, float]) -> bool:
    """
    Attempts to convert a string to a float to check whether it is numeric.
    This is not the same as :py:meth:`str::isnumeric()`, as this method
    essentially checks whether ``s`` contains something that looks like a
    number (int, float, scientific notation, etc).

    s: ``str``
        The string to check.

    :rtype: ``bool``

    :returns:
        `True` if the string is numeric; `False`, otherwise.
    """
    if isinstance(s, bool):
        return False
    if isinstance(s, int) or isinstance(s, float):
        return True
    if not isinstance(s, str):
        return False
    
    try:
        float(s)
        return True
    except:
        return False


def get_local_datasets() -> Iterable[LocalDataset]:
    """
    Opens the dataset directory and looks for locally available datasets.
    Locally available means datasets that were installed or created manually.
    A dataset is only considered to be locally available if it has a `manifest.json`.

    :rtype: ``Iterable[LocalDataset]``
    """
    for d in scandir(path=str(DATASETS_DIR.resolve())):
        if d.is_dir():
            if d.name == '_default':
                continue

            manifest = DATASETS_DIR.joinpath(f'./{d.name}/manifest.json')
            if manifest.exists():
                with open(file=str(manifest), mode='r', encoding='utf-8') as fp:
                    manifest: LocalDataset = load(fp=fp)
                    yield manifest



def get_known_datasets(use_local_file: bool=False) -> list[KnownDataset]:
    """
    Reads the file :py:data:`KNOWN_DATASETS_FILE` to obtain a list of known datasets.

    use_local_file: ``bool``
        If true, will attempt to read the known datasets from a local file, instead
        of the online file. This is only used during development.

    :rtype: ``list[KnownDataset]``
    """
    if use_local_file:
        with open(file=str(DATASETS_DIR.joinpath('./known-datasets.json')), mode='r', encoding='utf-8') as fp:
            return load(fp=fp)
    return loads(urlopen(url=KNOWN_DATASETS_FILE).read().decode('utf-8'))


def format_file_size(num_bytes: int, digits: int=2) -> str:
    """
    Formats bytes into a string with a suffix for bytes, kilobite, etc.
    The number of bytes in the prefix is less than 1000 and may be rounded
    to two decimals. For example: 780 B, 1.22 KB, 43 GB, etc. Does NOT use
    SI-2 suffixes as that would be non-sensical (e.g., what is 1.27 GiB?).

    num_bytes: ``int``
        Unsigned integer with amount of bytes.
    
    digits: ``int``
        The number of digits for rounding. If set to `0`, the rounded value
        is cast to integer.
    
    :returns:
        The size in bytes, formatted.
    """
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(num_bytes)

    i = 0
    while size > 1e3:
        size /= 1e3
        i += 1
    
    res = round(number=size, ndigits=digits)
    if digits == 0 or i == 0: # Still bytes
        res = int(res)
    
    return f'{res} {suffixes[i]}'
