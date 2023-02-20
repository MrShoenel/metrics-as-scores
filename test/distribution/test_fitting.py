import numpy as np
from re import escape
from pytest import raises
from scipy.stats._distn_infrastructure import rv_discrete
from scipy.stats._continuous_distns import norm, norm_gen
from metrics_as_scores.distribution.fitting import StatisticalTest, Fitter, FitterPymoo

def test_StatisticalTest():
    rng = np.random.default_rng(seed=1)
    st = StatisticalTest(
        data1=rng.normal(loc=5, scale=2, size=500),
        cdf=lambda x: norm.cdf(x=x, loc=5, scale=2),
        ppf_or_data2=lambda q: norm.ppf(q=q, loc=5, scale=2))
    
    # Let's assert that the (de-)serialization works.
    as_dict = dict(st)
    from_dict = StatisticalTest.from_dict(d=as_dict, key_prefix='')
    assert len(from_dict['tests']) == len(st.tests)


    # Let's test it throws if data2 is not callable and not array:
    with raises(Exception, match='must be 1d np.array or callable'):
        StatisticalTest(
            data1=rng.normal(loc=5, scale=2, size=500),
            cdf=lambda x: norm.cdf(x=x, loc=5, scale=2),
            ppf_or_data2='this will throw')
    

    st = StatisticalTest(data1=rng.normal(size=500), ppf_or_data2=rng.normal(size=500), max_samples=499, cdf=lambda x: norm.cdf(x=x, loc=0.0, scale=1.0))



def test_Fitter():
    with raises(Exception, match='The distribution given is not continuous or discrete random variable'):
        Fitter(dist=str)
    
    class Bla(rv_discrete):
        pass

    with raises(Exception, match='Cannot fit discrete distribution "Bla" because no practical ranges exist.'):
        Fitter(dist=Bla)


def test_FitterPymoo():
    with raises(Exception, match='The given object BLA is not a subclass of rv_generic.'):
        FitterPymoo(dist='BLA')
    
    class Bla(rv_discrete):
        pass

    with raises(Exception, match='No Pymoo problem available to optimize the random variable'):
        FitterPymoo(dist=Bla)
    

    # Test re-sampling:
    use_loc = 5.5
    use_scale = 2.2
    rng = np.random.default_rng(seed=1337)
    data = rng.normal(loc=use_loc, scale=use_scale, size=20_000)

    import warnings
    warnings.filterwarnings("error")
    with raises(Exception, match=escape("Fitting continuous random variable using variable's fit()-method")):
        fitter = FitterPymoo(dist=norm_gen)
    
    warnings.simplefilter("ignore")
    fitter = FitterPymoo(dist=norm_gen)
    res = fitter.fit(data=data, max_samples=19_900)
    
    warnings.resetwarnings()

    assert abs(res['loc'] - use_loc) <= 2e-2
    assert abs(res['scale'] - use_scale) <= 8e-3
