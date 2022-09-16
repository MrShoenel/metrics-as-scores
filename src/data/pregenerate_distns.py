from os import getcwd
from sys import path
from typing import Any, Union
from nptyping import Float, NDArray, Shape
path.append(getcwd())


import pandas as pd
import numpy as np
from gc import collect
from numpy.random import default_rng
from pickle import dump
from joblib import Parallel, delayed
from src.distribution.fitting import FitterPymoo, StatisticalTest
from src.distribution.fitting import Continuous_RVs, Discrete_RVs
from src.data.metrics import MetricID
from src.distribution.distribution import DensityFunc, DistTransform, Distribution
from sklearn.model_selection import ParameterGrid
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from pickle import dump

# import sys
# import traceback
# my_row = None
# def log_except_hook(*exc_info):
#     global my_row
#     text = "".join(traceback.format_exception(*exc_info()))
#     with open('./results/ex.pickle', 'wb') as f:
#         from pickle import dump
#         dump(my_row, f)
# sys.excepthook = log_except_hook



Continuous_RVs_dict: dict[str, type[rv_continuous]] = { x: y for (x, y) in zip(map(lambda rv: type(rv).__name__, Continuous_RVs), map(lambda rv: type(rv), Continuous_RVs)) }
Discrete_RVs_dict: dict[str, type[rv_discrete]] = { x: y for (x, y) in zip(map(lambda rv: type(rv).__name__, Discrete_RVs), map(lambda rv: type(rv), Discrete_RVs)) }





def generate_parametric_fits(distr_csv: str='csv/metrics.csv', dist_transform: DistTransform=DistTransform.NONE) -> dict[str, DensityFunc]:
    metrics_discrete_df = pd.read_csv('./files/metrics-discrete.csv')
    metrics_discrete = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_discrete_df.Metric), metrics_discrete_df.Discrete) }

    domains = ['__ALL__'] + Distribution.domains()

    # The thinking is this: To each data series we can always fit a continuous distribution,
    # whether it's discrete or continuous data. The same is not true the other way round, i.e.,
    # we should not fit a discrete distribution if the data is known to be continuous.
    # Therefore, we do the following:
    # - Regardless of the data, always attempt to fit a continuous RV
    # - For all discrete data, also attempt to fit a discrete RV
    #
    # That means that for discrete data, we will have to kinds of fitted RVs.
    # Also, when fitting a continuous RV to discrete data, we will add jitter to the data.
    param_grid = { 'domain': domains, 'metric': list(map(lambda m: m.name, MetricID)), 'rv': list(Continuous_RVs_dict.keys()), 'type': ['continuous'] }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))
    param_grid = { 'domain': domains, 'metric': list([x[0].name for x in metrics_discrete.items() if x[1]]), 'rv': list(Discrete_RVs_dict.keys()), 'type': ['discrete'] }
    expanded_grid = pd.concat([expanded_grid, pd.DataFrame(ParameterGrid(param_grid=param_grid))])

    def fit(grid_idx: int, row) -> dict[str, Any]:
        import sys
        if not sys.warnoptions:
            import warnings
            warnings.simplefilter("ignore")

        domain = row.domain
        metric_id = MetricID[row.metric]
        fit_continuous = row.type == 'continuous'
        RV: type[Union[rv_continuous, rv_discrete]] = None
        if fit_continuous:
            RV = Continuous_RVs_dict[row.rv]
        else:
            RV = Discrete_RVs_dict[row.rv]

        unique_vals = metrics_discrete[metric_id] and fit_continuous
        distr = Distribution(df=pd.read_csv(distr_csv))
        data = distr.data(metric_id=metric_id, unique_vals=unique_vals, domain=domain)
        transform_value, data = Distribution.transform(data=data, dist_transform=dist_transform)
        del distr
        collect() # GC

        ret_dict = dict(
            grid_idx = grid_idx,
            dist_transform = dist_transform,
            transform_value = transform_value,
            params = None, stat_tests = None)
        ret_dict.update(row.to_dict())

        try:
            fitter = FitterPymoo(dist=RV)
            params = fitter.fit(data=data, minimize_seeds=[1_3_3_7, 0xdeadbeef], verbose=False)
            params_tuple = tuple(params.values())

            ret_dict.update(dict(params = params))

            dist = RV()
            def temp_cdf(x: NDArray[Shape["*"], Float]):
                x = np.asarray(x).copy()
                min_ = np.min(data)
                max_ = np.max(data)
                x[x < min_] = min_
                x[x > max_] = max_
                return dist.cdf(*(x, *params_tuple))
            temp_ppf = lambda x: dist.ppf(*(x, *params_tuple))

            # BUG: https://github.com/scipy/scipy/issues/17026
            max_samples = 500 if row.rv == 'ncf_gen' else 25_000
            data_st = data if not unique_vals else np.rint(data) # Remove jitter for test
            st = StatisticalTest(data1=data_st, cdf=temp_cdf, ppf_or_data2=temp_ppf, max_samples=max_samples)
            ret_dict.update(dict(stat_tests = dict(st)))
        except Exception as e:
            print(e)
            # Do nothing at the moment, and allow returning a dict without params and stat_tests
            pass

    # def wrapper(*args):
    #     try:
    #         return fit(*args)
    #     except Exception as e:
    #         print(e)
    #     return 123
    
    # row = expanded_grid.iloc[1,]
    # row.domain = 'testing' # 'IDE' # 'database' # 'parsers/generators/make'
    # row.metric = 'CE' # 'DIT'
    # row.type = 'continuous'
    # row.rv = 'kstwo_gen' # 'ncf_gen'
    # res = fit(1, row)

    rng = default_rng(seed=76543210)
    indexes = rng.choice(a=list(range(len(expanded_grid))), replace=False, size=len(expanded_grid))
    from tqdm import tqdm
    res = Parallel(n_jobs=-1)(delayed(fit)(i, expanded_grid.iloc[i,]) for i in tqdm(indexes))

    return res


if __name__ == '__main__':
    #from scipy.stats._continuous_distns import ncf_gen
    #use_x = np.linspace(start=1e-16, stop=1. - (1e-16), num=50_000)
    #params = (0.13826005075650766, 5.269495454730334, 5.734469579885159, 1.0630077496162628e-08, 0.03580180061763037)
    #dfn, dfd, nc, loc, scale = 27, 27, 0.416, 0, 1
    #dist = ncf_gen()
    #bla = dist.ppf(*(use_x, *(dfn, dfd, nc, loc, scale)))
    #bla = dist.ppf(*(use_x, *params))
    #print(bla)

    # from pickle import load
    # with open(file='./results/row_15269.pickle', mode='rb') as f:
    #     results = load(f)
    #     print(results)

    #distr = Distribution(df=pd.read_csv('csv/metrics.csv'))
    #data = distr.data(MetricID.NOF, 'testing', unique_vals=True)
    #from scipy.stats._continuous_distns import exponpow_gen
    #params = exponpow_gen().fit(data)


    for dist_transform in list(DistTransform):
        result = generate_parametric_fits(distr_csv='csv/metrics.csv', dist_transform=dist_transform)
        with open(file=f'./results/pregnerate_distns_{dist_transform.name}.pickle', mode='wb') as f:
            dump(result, f)
