from toml import load
from pathlib import Path

proj_dir = Path(__file__).parent.parent.parent
__version__ = load(proj_dir.joinpath('./pyproject.toml')).get('version')
