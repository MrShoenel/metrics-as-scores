import numpy as np
from pytest import raises
from scipy.stats._continuous_distns import norm
from metrics_as_scores.distribution.fitting import StatisticalTest

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
