import numpy as np
import warnings
from pytest import raises
from numpy.random import default_rng
from scipy.stats._continuous_distns import norm_gen
from scipy.stats._discrete_distns import poisson_gen
from metrics_as_scores.distribution.distribution import DistTransform, Density, KDE_integrate, KDE_approx, Empirical, Empirical_discrete, Parametric, Parametric_discrete
from metrics_as_scores.distribution.fitting import StatisticalTest
from metrics_as_scores.distribution.fitting import Fitter
from metrics_as_scores.tools.funcs import flatten_dict


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


def test_Empirical():
    rng = default_rng(seed=1337)
    data = rng.normal(loc=0.0, scale=1.0, size=5_000)
    dens = Empirical(data=data, compute_ranges=True, qtype='q', context='c')

    # Make sure everything works:
    assert isinstance(dens.pdf(0.0)[0], float)
    assert (dens.cdf(0) - dens.cdf(-1e3) - 0.5) <= 2e-3
    assert abs(dens.ppf(0.5)) <= 5e-3


def test_Empirical_discrete():
    rng = default_rng(seed=1337)

    # Should not work with continuous:
    data = rng.normal(size=1_000)
    with raises(Exception, match='The data does not appear to be integral.'):
        Empirical_discrete(data=data)

    # Doesn't raise, but is invalid:
    dens = Empirical_discrete(data=np.array([float('nan')]))
    assert dens.is_fit == False
    dens = Empirical_discrete.unfitted(dist_transform=DistTransform.NONE)
    assert dens.is_fit == False
    # Its CDF/PPF only return NaNs:
    assert np.all(np.isnan(dens.cdf(rng.normal(size=100))))
    assert np.all(np.isnan(dens.ppf(rng.normal(size=100))))

    # Let's make it valid:
    data = rng.poisson(size=500)
    dens = Empirical_discrete(data=data)
    assert dens.is_fit
    # Let's make sure the PMF sums to 1:
    probs = np.array([dens.pmf(x) for x in range(np.min(data), np.max(data) + 1)])
    assert abs(1.0 - np.sum(probs)) <= 1e-12
    assert dens.pmf(np.min(data) - 10) <= 1e-300 # technically zero


def test_Parametric():
    rng = default_rng(seed=1337)
    data = rng.normal(loc=0.0, scale=1.0, size=5_000)
    norm = norm_gen()
    theta = norm.fit(data=data)

    st = StatisticalTest(data1=data, cdf=lambda x: norm.cdf(x, *theta), ppf_or_data2=lambda p: norm.ppf(p, *theta))
    with raises(Exception, match=f'This class requires a dictionary of statistical tests, not an instance of {StatisticalTest.__name__}.'):
        Parametric(norm_gen(), dist_params=(0.0, 1.0), range=(-1e3, 1e3), stat_tests=st, compute_ranges=True)
    with raises(Exception, match=f'Key not allowed'):
        Parametric(norm_gen(), dist_params=(0.0, 1.0), range=(-1e3, 1e3), stat_tests=dict(st), compute_ranges=True)
    with raises(Exception, match=f'Value for key "FOO_BAR_pval" is not numeric.'):
        temp_st = StatisticalTest.from_dict(d=flatten_dict(dict(st)), key_prefix='tests_')
        temp_st['FOO_BAR_pval'] = 'bla'
        Parametric(norm_gen(), dist_params=(0.0, 1.0), range=(-1e3, 1e3), stat_tests=temp_st, compute_ranges=True)
    
    dens = Parametric(norm_gen(), dist_params=(0.0, 1.0), range=(-1e3, 1e3), stat_tests=flatten_dict(dict(st)['tests']), compute_ranges=True)
    assert dens.dist_name == 'norm_gen'
    assert dens.is_fit
    assert dens.use_stat_test == 'ks_2samp_jittered'
    assert np.allclose(dens.cdf(rng.normal(loc=-1e4, size=100)), np.zeros(shape=(100,))) # Those are out of range
    assert np.allclose(dens.cdf(rng.normal(loc= 1e4, size=100)), np.ones(shape=(100,))) # Those are out of range
    temp = dens.ppf(rng.uniform(low=0.1, high=0.9, size=100))
    assert np.all((temp > -50) & (temp < 50))

    assert isinstance(dens.pval, float) and not np.isnan(dens.pval)
    assert isinstance(dens.stat, float) and not np.isnan(dens.stat)
    dens.use_stat_test = 'epps_singleton_2samp_jittered'
    assert isinstance(dens.pval, float) and not np.isnan(dens.pval)
    assert isinstance(dens.stat, float) and not np.isnan(dens.stat)


    # Test unfit:
    dens = Parametric.unfitted(dist_transform=DistTransform.NONE)
    assert dens.is_fit == False
    temp = dens.practical_domain
    assert temp[0] == 0 and temp[1] == 0
    temp = dens.practical_range_pdf
    assert temp[0] == 0 and temp[1] == 0

    assert np.allclose(dens.pdf(rng.normal(size=100)), np.zeros(shape=(100,)))
    assert np.allclose(dens.cdf(rng.normal(size=100)), np.zeros(shape=(100,)))
    assert np.allclose(dens.ppf(rng.uniform(low=0.0, high=1.0, size=100)), np.zeros(shape=(100,)))

    with raises(Exception):
        dens.pval
    with raises(Exception):
        dens.stat
    
    dens.compute_practical_domain()


def test_Parametric_discrete():
    rng = default_rng(seed=1)
    data = rng.poisson(size=50)
    warnings.filterwarnings("ignore", message="divide by zero encountered") 
    fitter = Fitter(poisson_gen)
    warnings.simplefilter('always')
    params = fitter.fit(data=data)
    theta = tuple(params.values())
    inst = poisson_gen()

    # Make a fit one:
    st = StatisticalTest(data1=data, cdf=lambda x: inst.cdf(x, *theta), ppf_or_data2=lambda p: inst.ppf(p, *theta))
    dens = Parametric_discrete(dist=poisson_gen(), dist_params=theta, range=(np.min(data), np.max(data)), stat_tests=flatten_dict(dict(st)['tests']))

    # Let's make sure the PMF sums to ~1:
    # Since it's a smooth parametric distribution, we gotta widen the range
    probs = np.array([dens.pmf(x) for x in range(np.min(data)-20, np.max(data) + 20)])
    assert abs(1.0 - np.sum(probs)) <= 1e-12
    assert dens.pmf(np.min(data) - 20) <= 1e-300 # technically zero


    # Unfit -> dist_params=None
    dens = Parametric_discrete.unfitted(dist_transform=DistTransform.NONE)
    assert np.allclose(dens.pdf(rng.normal(size=100)), np.zeros(shape=(100,)))
    assert np.allclose(dens.cdf(rng.normal(size=100)), np.zeros(shape=(100,)))
    assert np.allclose(dens.ppf(rng.uniform(low=0.0, high=1.0, size=100)), np.zeros(shape=(100,)))
