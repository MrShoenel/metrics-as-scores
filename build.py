from pathlib import Path
from shutil import copyfile

root_dir = Path(__file__).parent
mas_dir = root_dir.joinpath('./src/metrics_as_scores')
pyproject_file = root_dir.joinpath('./pyproject.toml')

copyfile(src=str(pyproject_file), dst=str(mas_dir.joinpath('./pyproject.toml')))
