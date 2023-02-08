"""
This is an extra module that holds functions globally, such that we
can exploit multiprocessing effortlessly.
"""

import numpy as np
from pickle import dump
from typing import Any, TypedDict, Union
from nptyping import Float, NDArray, Shape
from metrics_as_scores.distribution.distribution import DistTransform, Dataset
from metrics_as_scores.distribution.fitting import Continuous_RVs, Discrete_RVs, Fitter, StatisticalTest, StatisticalTestJson
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from timeit import default_timer as timer



Continuous_RVs_dict: dict[str, type[rv_continuous]] = { x: y for (x, y) in zip(map(lambda rv: type(rv).__name__, Continuous_RVs), map(lambda rv: type(rv), Continuous_RVs)) }
Discrete_RVs_dict: dict[str, type[rv_discrete]] = { x: y for (x, y) in zip(map(lambda rv: type(rv).__name__, Discrete_RVs), map(lambda rv: type(rv), Discrete_RVs)) }



def get_data_tuple(dist: Dataset, metric_id: MetricID, dist_transform: DistTransform, continuous_transform: bool=True) -> list[tuple[str, NDArray[Shape["*"], Float]]]:
    l = []
    for dom in Dataset.domains(include_all_domain=True):
        for unique_vals in [True, False]:
            data = dist.data(metric_id=metric_id, domain=(None if dom == '__ALL__' else dom), unique_vals=unique_vals, sub_sample=25_000)
            transform_value, data = Dataset.transform(data=data, dist_transform=dist_transform, continuous_value=continuous_transform)
            key = f"{dom}_{metric_id.name}{('_u' if unique_vals else '')}"
            l.append((key, data, transform_value))
    return l


def fit(grid_idx: int, row, metrics_discrete: dict[MetricID, bool], the_data: NDArray[Shape["*"], Float], the_data_unique: NDArray[Shape["*"], Float], transform_value: Union[float, None], dist_transform: DistTransform, write_temporary_results: bool=True) -> dict[str, Any]:
    start = timer()
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    metric_id = MetricID[row.metric]
    fit_continuous = row.type == 'continuous'
    RV: type[Union[rv_continuous, rv_discrete]] = None
    if fit_continuous:
        RV = Continuous_RVs_dict[row.rv]
    else:
        RV = Discrete_RVs_dict[row.rv]

    unique_vals = metrics_discrete[metric_id] and fit_continuous
    data = the_data_unique if unique_vals else the_data

    ret_dict = dict(
        grid_idx = grid_idx,
        dist_transform = dist_transform.name,
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

        data_st = data if not unique_vals else np.rint(data) # Remove jitter for test
        st = StatisticalTest(data1=data_st, cdf=temp_cdf, ppf_or_data2=temp_ppf, max_samples=25_000)
        ret_dict.update(dict(stat_tests = dict(st)))
    except Exception as e:
        print(e)
        # Do nothing at the moment, and allow returning a dict without params and stat_tests
        pass
    finally:
        if write_temporary_results:
            end = timer() - start
            print(f'DONE! it took {format(end, "0>5.0f")} seconds ({row.type}), [{row.rv}]')
            with open(f'./results/temp/{grid_idx}_{format(end, "0>5.0f")}', 'wb') as f:
                dump(ret_dict, f)

    return ret_dict
