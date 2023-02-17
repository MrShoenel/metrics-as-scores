import numpy as np
from numpy.random import default_rng
from metrics_as_scores.distribution.fitting import FitterPymoo
from scipy.stats._discrete_distns import bernoulli_gen


def test_Fit_poisson_gen():
    fp = FitterPymoo(dist=bernoulli_gen)
    rng = default_rng(seed=1337)
    data = rng.binomial(n=1, p=0.5, size=50)

    res_dict = fp.fit(data=data)
    assert isinstance(res_dict, dict)
    assert 'p' in res_dict
    assert isinstance(res_dict['p'], float)
    # At least roughly ;)
    # Usually converges to ~0.46
    assert res_dict['p'] > .4 and res_dict['p'] < .6