from nptyping import NDArray
from numpy import abs, cumsum, linspace, max, min, square, sum, vectorize
from typing import Callable


def nonlinspace(start: float, stop: float, num: int, func: Callable[[float], float]=lambda x: 1. - .9 * square(x)) -> NDArray:
    if abs(stop - start) < 1e20:
        return linspace(start=start, stop=stop, num=num)
    func = vectorize(func)
    step_lens = func(linspace(start=-1., stop=1., num=num))
    x_prime = step_lens / sum(step_lens)
    temp = cumsum(x_prime * (stop - start))
    temp -= min(temp)
    return temp / max(temp) * (stop-start) + start
