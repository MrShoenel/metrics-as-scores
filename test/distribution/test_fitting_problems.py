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


def test_init_other_problems():
    from metrics_as_scores.distribution.fitting_problems import Fit_betabinom_gen, Fit_binom_gen, Fit_boltzmann_gen, Fit_dlaplace_gen, Fit_geom_gen, Fit_hypergeom_gen, Fit_logser_gen, Fit_nbinom_gen, Fit_nchypergeom_fisher_gen, Fit_nchypergeom_wallenius_gen, Fit_nhypergeom_gen, Fit_planck_gen, Fit_poisson_gen, Fit_randint_gen, Fit_skellam_gen, Fit_yulesimon_gen, Fit_zipf_gen, Fit_zipfian_gen

    data = np.rint(np.random.normal(loc=10.0, scale=5.0, size=25))
    
    Fit_betabinom_gen(data=data)
    Fit_binom_gen(data=data)
    Fit_boltzmann_gen(data=data)
    Fit_dlaplace_gen(data=data)
    Fit_geom_gen(data=data)
    Fit_logser_gen(data=data)
    Fit_nbinom_gen(data=data)
    Fit_nchypergeom_wallenius_gen(data=data)
    Fit_nhypergeom_gen(data=data)
    Fit_planck_gen(data=data)
    Fit_poisson_gen(data=data)
    Fit_randint_gen(data=data)
    Fit_skellam_gen(data=data)
    Fit_yulesimon_gen(data=data)
    Fit_zipf_gen(data=data)
    Fit_zipfian_gen(data=data)


    # Also call _evaluate of some classes:
    f = Fit_hypergeom_gen(data=data)
    out = dict()
    f._evaluate(X=dict(
        loc = np.mean(data),
        M = 10,
        N = 10,
        n = 0
    ), out=out)
    assert isinstance(out['F'], float)
    assert isinstance(out['G'], np.ndarray)


    f = Fit_nchypergeom_fisher_gen(data=data)
    out = dict()
    f._evaluate(X=dict(
        loc = np.mean(data),
        M = 10,
        N = 10,
        n = 10,
        odds = 1.0
    ), out=out)
    assert isinstance(out['F'], float)
    assert isinstance(out['G'], np.ndarray)
