from os import getcwd
from os.path import join
from sys import path
from pathlib import Path
path.append(f'{Path(join(getcwd(), "src")).resolve()}')

from typing import Union
from metrics_as_scores.distribution.distribution import Empirical, KDE_approx, Parametric
from metrics_as_scores.tools.lazy import SelfResetLazy


cdfs: dict[str, SelfResetLazy[dict[str, Union[Empirical, KDE_approx, Parametric]]]] = None
