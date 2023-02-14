from sys import path
from pathlib import Path
from metrics_as_scores.__version__ import IS_MAS_LOCAL
MAS_DIR = Path(__file__).resolve().parent
path.append(str(MAS_DIR.resolve()))
path.append(str(MAS_DIR.parent.resolve()))

DATASETS_DIR: Path = (MAS_DIR.parent.parent if IS_MAS_LOCAL else MAS_DIR).joinpath('./datasets')
"""
This is the directory that holds the downloaded and manually created datasets.
This directory is in the project's root if Metrics As Scores was cloned and
the full project is present.
Otherwise, when the Metrics As Scores is installed from PyPI as a package, we
store downloaded and own datasets in the datasets-folder within the source.
Have a look at :py:const:`IS_MAS_LOCAL`.

:meta hide-value:
"""
if not DATASETS_DIR.exists():
    DATASETS_DIR.mkdir(exist_ok=False)
