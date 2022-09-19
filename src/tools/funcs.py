from collections.abc import MutableMapping
from nptyping import NDArray
from numpy import abs, cumsum, linspace, max, min, square, sum, vectorize
from typing import Any, Callable


def nonlinspace(start: float, stop: float, num: int, func: Callable[[float], float]=lambda x: 1. - .9 * square(x)) -> NDArray:
    if abs(stop - start) < 1e20:
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
