from sys import path
from pathlib import Path
from metrics_as_scores.__version__ import IS_MAS_LOCAL
MAS_DIR = Path(__file__).resolve().parent
path.append(str(MAS_DIR.resolve()))
path.append(str(MAS_DIR.parent.resolve()))

DATASETS_DIR = (MAS_DIR.parent.parent if IS_MAS_LOCAL else MAS_DIR).joinpath('./datasets')
if not DATASETS_DIR.exists():
    DATASETS_DIR.mkdir(exist_ok=False)
