import numpy as np
from pytest import raises
from numpy.random import default_rng
from scipy.stats._continuous_distns import norm_gen
from metrics_as_scores.distribution.distribution import DistTransform, Density, KDE_integrate, KDE_approx


def test_Density():
    rng = default_rng(seed=1337)
    data = rng.normal(loc=0.0, scale=1.0, size=5_000)
    norm = norm_gen()
    theta = norm.fit(data=data)
    dens = Density(
        range=(np.min(data), np.max(data)), qtype='q', context='c',
        pdf=lambda x: norm.pdf(x=x, *theta),
        cdf=lambda x: norm.cdf(x=x, *theta))

    with raises(NotImplementedError):
        dens.ppf(0.5)
    
    assert dens.qtype == 'q' and dens.context == 'c'
    assert dens.ideal_value is None and dens.transform_value is None
    assert dens.dist_transform == DistTransform.NONE


def test_KDE_integrate():
    rng = default_rng(seed=1337)
    data = rng.normal(loc=0.0, scale=1.0, size=5_000)
    dens = KDE_integrate(data=data).init_ppf(cdf_samples=10)

    assert (dens.cdf(0) - dens.cdf(-1e3) - 0.5) <= 2e-3


def test_KDE_approx():
    use_loc = 3.5
    rng = default_rng(seed=1337)
    data = rng.normal(loc=use_loc, scale=2.7, size=5_000)

    dens = KDE_approx(data=data, resample_samples=10_000, compute_ranges=True, qtype='', context='')
    assert dens.cdf(-20) <= 1e-10
    assert (1.0 - dens.cdf(20)) <= 1e-10

    pd = dens.practical_domain
    assert pd[0] >= -100 and pd[1] <= 100

    # Make sure we can get a p-value/statistic:
    assert isinstance(dens.pval, float) and isinstance(dens.stat, float)

    # Let's call the density, which returns its CDF:
    assert abs(0.5 - dens(use_loc)) <= 2e-3
