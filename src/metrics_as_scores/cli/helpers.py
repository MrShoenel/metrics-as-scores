"""
This module contains constants and helper functionss that are commonly
used in any of the CLI workflows.
"""

from os import scandir
from pathlib import Path
from typing import Iterable, Union
from json import load, loads
from urllib.request import urlopen
from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.distribution.distribution import LocalDataset, KnownDataset, Parametric, Parametric_discrete, Empirical, Empirical_discrete, KDE_approx, DistTransform
from itertools import product
from strenum import StrEnum


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



def required_files_folders_local_dataset(local_ds_id: str) -> tuple[list[Path], list[Path]]:
    """
    For a given :py:class:`LocalDataset`, returns lists of directories and files
    that must be present in order for the local dataset to be valid. These lists
    of directories and files must be checked when, e.g., the dataset is bundled,
    or when the web application is instructed to use a local dataset.

    local_ds_id: ``str``
        The ID of a local dataset to check files for.
    
    rtype: ``tuple[list[Path], list[Path]]``

    :return:
        A list of paths to required folders and a list of paths to required files.
    """

    ds_dir = DATASETS_DIR.joinpath(f'./{local_ds_id}')

    dir_densities = ds_dir.joinpath(f'./densities')
    dir_fits = ds_dir.joinpath(f'./fits')
    dir_stests = ds_dir.joinpath(f'./stat-tests')
    dir_web = ds_dir.joinpath(f'./web')

    dirs_required = [
        dir_densities,
        dir_fits,
        dir_stests,
        dir_web
    ]

    files_required = list([
        dir_densities.joinpath(f'./densities_{perm[0].__name__}_{perm[1].name}.pickle')
        for perm in product([
            Parametric, Parametric_discrete, Empirical, Empirical_discrete, KDE_approx
        ], list(DistTransform))
    ]) + list([
        dir_fits.joinpath(f'./pregen_distns_{dt.name}.pickle') for dt in list(DistTransform)
    ]) + list([
        dir_fits.joinpath(f'./pregen_distns_{dt.name}.csv') for dt in list(DistTransform)
    ]) + list([
        dir_stests.joinpath(f'./{file}.csv') for file in ['anova', 'ks2samp', 'tukeyhsd', 'kruskal']
    ]) + list([
        dir_web.joinpath(f'./{file}.html') for file in ['about', 'references']
    ]) + list([
        ds_dir.joinpath(f'./{file}') for file in [
            'About.pdf',
            'manifest.json',
            'org-data.csv',
            'refs.bib'
        ]
    ])

    return dirs_required, files_required


class PathStatus(StrEnum):
    """
    This is an enumeration of statuses for :py:class:`Path` objects that point
    to directories or files.
    """

    OK = 'OK'
    """The file or directory exists and is of correct type."""

    DOESNT_EXIST = 'Does not exist'
    """The file or directory does not exist."""

    NOT_A_DIRECTORY = 'Not a directory'
    """The Path exists, but is not a directory."""

    NOT_A_FILE = 'Not a file'
    """The Path exists, but is not a file."""


def validate_local_dataset_files(dirs: list[Path], files: list[Path]) -> tuple[dict[Path, PathStatus], dict[Path, PathStatus]]:
    """
    Takes two lists, one of paths of directories, and one of paths to files of a
    local dataset from :py:meth:`required_files_folders_local_dataset()`. Then for
    each item on each list, checks whether it exists and is of correct type, then
    associates a :py:class:`PathStatus` with each.

    dirs: ``list[Path]``
        A list of paths to directories needed in a local dataset.
    
    files: ``list[Path]``
        A list of paths to files needed in a local dataset.
    
    rtype: ``tuple[dict[Path, PathStatus], dict[Path, PathStatus]]``

    :return:
        Transforms either list into a dictionary using the original path as key
        and a :py:class:`PathStatus` as value. Then returns both dictionaries.
    """
    dirs_dict: dict[Path, PathStatus] = dict()
    for d in dirs:
        if not d.exists():
            dirs_dict[d] = PathStatus.DOESNT_EXIST
        elif not d.is_dir():
            dirs_dict[d] = PathStatus.NOT_A_DIRECTORY
        else:
            dirs_dict[d] = PathStatus.OK
    
    files_dict: dict[Path, PathStatus] = dict()
    for f in files:
        if not f.exists():
            files_dict[f] = PathStatus.DOESNT_EXIST
        elif not f.is_file():
            files_dict[f] = PathStatus.NOT_A_FILE
        else:
            files_dict[f] = PathStatus.OK
    
    return dirs_dict, files_dict


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
