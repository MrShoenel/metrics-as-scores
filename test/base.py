from os import getcwd
from os.path import join
from sys import path
from pathlib import Path
path.append(f'{Path(join(getcwd(), "src")).resolve()}')