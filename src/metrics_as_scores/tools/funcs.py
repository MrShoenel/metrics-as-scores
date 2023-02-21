"""
This module contains few functions that are required in multiple other modules
and classes of Metrics As Scores.
"""

import numpy as np
from re import split
from collections.abc import MutableMapping
from nptyping import Float, NDArray, Shape
from numpy import abs, cumsum, linspace, max, min, square, sum, vectorize, interp
from typing import Any, Callable, Union


def nonlinspace(start: float, stop: float, num: int, func: Callable[[float], float]=lambda x: 1. - .9 * square(x)) -> NDArray:
    """
    Used to create a non-linear space. This is useful to sample with greater detail
    in some place, and with lesser detail in another.

    start: ``float``
        The start of the space.
    
    stop: ``float``
        The stop (end) of the space.
    
    num: ``int``
        The number of samples.
    
    func: ``Callable[[float], float]``
        A function that, given a linear `x`, creates a non-linear `y`. The default
        is :code:`lambda x: 1. - .9 * square(x)`.
    """
    if abs(stop - start) < 1e-20:
        return linspace(start=start, stop=stop, num=num)
    func = vectorize(func)
    step_lens = func(linspace(start=-1., stop=1., num=num))
    x_prime = step_lens / sum(step_lens)
    temp = cumsum(x_prime * (stop - start))
    temp -= min(temp)
    return temp / max(temp) * (stop-start) + start


def flatten_dict(d: dict[str, Any], parent_key: str='', sep: str='_') -> dict[str, Any]:
    """
    Recursively flatten a dictionary, creating merged key names.
    Courtesy of https://stackoverflow.com/a/6027615/1785141
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(d=v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Interpolator:
    """
    Helper class used for transforming and interpolating a CDF into a PPF.
    """
    def __init__(self, xp: NDArray[Shape["*"], Float], fp: NDArray[Shape["*"], Float], left: float=None, right: float=None) -> None:
        self.xp = xp
        self.fp = fp
        self.left = left
        self.right = right
    
    def __call__(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return interp(x=x, xp=self.xp, fp=self.fp, left=self.left, right=self.right)


def cdf_to_ppf(cdf: Callable[[float], float], x: NDArray[Shape["*"], Float], cdf_samples: int=5_000, y_left: float=None, y_right: float=None) -> Union[Interpolator, Callable[[float], float]]:
    """
    Adapted implementation from :meth:`statsmodels.distributions.empirical_distribution.monotone_fn_inverter()`
    that handles out-of-bounds values explicitly. Also, we assume that ``fn`` is vectorized.
    """
    x_vals = np.linspace(start=np.min(x), stop=np.max(x), num=cdf_samples)
    y_vals = cdf(x_vals) # x is sorted, then so is y if fn is monotone increasing (which it should be)

    return Interpolator(xp=y_vals, fp=x_vals, left=y_left, right=y_right)


def natsort(s: str) -> int:
    """
    Natural string sorting.

    Courtesy of https://stackoverflow.com/a/16090640/1785141
    """
    return [int(t) if t.isdigit() else t.lower() for t in split(r'(\d+)', f'{s}')]
