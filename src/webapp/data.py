from typing import Union
from src.distribution.distribution import ECDF, KDECDF_approx, ParametricCDF
from src.tools.lazy import SelfResetLazy


cdfs: dict[str, SelfResetLazy[dict[str, Union[ECDF, KDECDF_approx, ParametricCDF]]]] = None
