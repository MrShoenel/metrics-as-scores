from typing import Union
from src.distribution.distribution import Empirical, KDE_approx, Parametric
from src.tools.lazy import SelfResetLazy


cdfs: dict[str, SelfResetLazy[dict[str, Union[Empirical, KDE_approx, Parametric]]]] = None
