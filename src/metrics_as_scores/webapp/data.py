"""
The purpose of this module is to exchange data only once, similar to
a singleton pattern. When the application loads, the app-hooks are
called first. Those will load the correct dataset and set values to
the below properties *only once* during application lifecycle. Then,
the web application can read these values.
"""
from pathlib import Path
from typing import Union
from metrics_as_scores.distribution.distribution import Empirical, KDE_approx, Parametric, Dataset
from metrics_as_scores.tools.lazy import SelfResetLazy


cdfs: dict[str, SelfResetLazy[dict[str, Union[Empirical, KDE_approx, Parametric]]]] = None
"""A dictionary of the dataset's densities.

:meta hide-value:
"""
ds: Dataset = None
"""The dataset itself, including its manifest.

:meta hide-value:"""
dataset_dir: Path = None
"""The absolute path to the dataset. Needed primarily to load its HTML-fragments.

:meta hide-value:"""
