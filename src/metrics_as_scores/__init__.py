from sys import path
from pathlib import Path
mas_dir = Path(__file__).resolve().parent
path.append(str(mas_dir.resolve()))
path.append(str(mas_dir.parent.resolve()))
