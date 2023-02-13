from toml import load
from pathlib import Path

this_dir = Path(__file__).parent
proj_dir = Path(__file__).parent.parent.parent
pyproject_local = this_dir.joinpath('./pyproject.toml')
pyproject_root = proj_dir.joinpath('./pyproject.toml')

__version__: str = None
if pyproject_root.exists() and load(pyproject_root)['tool']['poetry']['name'] == 'metrics-as-scores':
    __version__ = load(pyproject_root)['tool']['poetry']['version']
elif pyproject_local.exists():
    __version__ = load(pyproject_local)['tool']['poetry']['version']
else:
    raise Exception('Cannot determine version.')

