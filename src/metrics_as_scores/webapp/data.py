from pathlib import Path
from typing import Union
from metrics_as_scores.distribution.distribution import Empirical, KDE_approx, Parametric, Dataset
from metrics_as_scores.tools.lazy import SelfResetLazy


cdfs: dict[str, SelfResetLazy[dict[str, Union[Empirical, KDE_approx, Parametric]]]] = None
ds: Dataset = None
dataset_dir: Path = None
