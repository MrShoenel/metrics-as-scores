from os import getcwd, cpu_count
from sys import path
from typing import Any
path.append(getcwd())


import numpy as np
import pandas as pd
from pickle import dump
from nptyping import Float, NDArray, Shape
from gc import collect
from tqdm import tqdm
from numpy.random import default_rng
from pickle import dump
from joblib import Parallel, delayed
from src.data.metrics import MetricID
from src.distribution.distribution import DistTransform, Dataset, Parametric, Parametric_discrete
from src.data.pregenerate_fit import fit, get_data_tuple, Continuous_RVs_dict, Discrete_RVs_dict
from sklearn.model_selection import ParameterGrid


metrics_discrete_df = pd.read_csv('./files/metrics-discrete.csv')
metrics_discrete = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_discrete_df.Metric), metrics_discrete_df.Discrete) }


def generate_parametric_fits(dist_transform: DistTransform, data_dict: dict[str, NDArray[Shape["*"], Float]], transform_values_dict: dict[str, float], data_discrete_dict: dict[str, NDArray[Shape["*"], Float]], transform_values_discrete_dict: dict[str, float]) -> list[dict[str, Any]]:
    # The thinking is this: To each data series we can always fit a continuous distribution,
    # whether it's discrete or continuous data. The same is not true the other way round, i.e.,
    # we should not fit a discrete distribution if the data is known to be continuous.
    # Therefore, we do the following:
    #
    # - Regardless of the data, always attempt to fit a continuous RV
    # - For all discrete data, also attempt to fit a discrete RV
    #
    # That means that for discrete data, we will have to kinds of fitted RVs.
    # Also, when fitting a continuous RV to discrete data, we will add jitter to the data.

    domains = Dataset.domains(include_all_domain=True)

    from scipy.stats._continuous_distns import norminvgauss_gen, gausshyper_gen, genhyperbolic_gen, geninvgauss_gen, invgauss_gen, studentized_range_gen
    from scipy.stats._discrete_distns import nhypergeom_gen, hypergeom_gen
    ignored_dists = list([rv.__name__ for rv in [norminvgauss_gen, gausshyper_gen, genhyperbolic_gen, geninvgauss_gen, invgauss_gen, studentized_range_gen, nhypergeom_gen, hypergeom_gen]])

    param_grid = { 'domain': domains, 'metric': list(map(lambda m: m.name, MetricID)), 'rv': list([x for x in Continuous_RVs_dict.keys() if x not in ignored_dists]), 'type': ['continuous'], 'dist_transform': [dist_transform] }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))
    param_grid = { 'domain': domains, 'metric': list([x[0].name for x in metrics_discrete.items() if x[1]]), 'rv': list([x for x in Discrete_RVs_dict.keys() if x not in ignored_dists]), 'type': ['discrete'], 'dist_transform': [dist_transform] }
    expanded_grid = pd.concat([expanded_grid, pd.DataFrame(ParameterGrid(param_grid=param_grid))])

    def get_datas(grid_idx) -> tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float], float]:
        row = expanded_grid.iloc[grid_idx,]
        key = f'{row.domain}_{row.metric}'
        key_u = f'{key}_u'
        discrete_discrete = MetricID[row.metric] in metrics_discrete.keys() and row.rv in Discrete_RVs_dict.keys()
        if discrete_discrete:
            # Fit a discrete RV to discrete data
            return (data_discrete_dict[key], data_discrete_dict[key_u], transform_values_discrete_dict[key])
        return (data_dict[key], data_dict[key_u], transform_values_dict[key])

    rng = default_rng(seed=76543210)
    indexes = rng.choice(a=list(range(len(expanded_grid))), replace=False, size=len(expanded_grid))
    res = Parallel(n_jobs=-1)(delayed(fit)(i, expanded_grid.iloc[i,], metrics_discrete, *get_datas(grid_idx=i), dist_transform) for i in tqdm(indexes))

    return res



if __name__ == '__main__':
    #df = pd.read_csv('./results/temp.csv')
    #bla = fits_to_mas_cdfs(df=df, dist_transform=DistTransform.EXPECTATION, use_continuous=False)
    #with open('./results/cdfs_ParametricCDF_discrete_EXPECTATION.pickle', 'wb') as f:
    #    dump(bla, f)
    #with open('./results/distns_NONE.pickle', 'rb') as f:
    #    from pickle import load
    #    bla = load(f)
    #    print(bla)

    print('Reading data file...')
    dist = Dataset(df=pd.read_csv('csv/metrics.csv'))

    for dist_transform in list(DistTransform):
        print(f'Parallel pre-processing of datasets for transform {dist_transform.name} ({dist_transform.value})...')

        use_metrics = list([m for m in MetricID if metrics_discrete[m]])
        res = Parallel(n_jobs=min(len(use_metrics), cpu_count()))(delayed(get_data_tuple)(dist, m, dist_transform, False) for m in tqdm(use_metrics))
        data_discrete_dict: dict[str, NDArray[Shape["*"], Float]] = dict([(item[0], item[1]) for sublist in res for item in sublist])
        transform_values_discrete_dict: dict[str, float] = dict([(item[0], item[2]) for sublist in res for item in sublist])
        del res
        collect()
        print(f'Having {len(list(data_discrete_dict.keys()))} pre-processed datasets (discrete).')
        
        use_metrics = list(MetricID)
        res = Parallel(n_jobs=min(len(use_metrics), cpu_count()))(delayed(get_data_tuple)(dist, m, dist_transform, True) for m in tqdm(use_metrics))
        data_dict: dict[str, NDArray[Shape["*"], Float]] = dict([(item[0], item[1]) for sublist in res for item in sublist])
        transform_values_dict: dict[str, float] = dict([(item[0], item[2]) for sublist in res for item in sublist])
        del res
        collect()
        print(f'Having {len(list(data_dict.keys()))} pre-processed datasets (continuous).')

        result = generate_parametric_fits(dist_transform=dist_transform, data_dict=data_dict, transform_values_dict=transform_values_dict, data_discrete_dict=data_discrete_dict, transform_values_discrete_dict=transform_values_discrete_dict)
        with open(file=f'./results/pregnerate_distns_{dist_transform.name}.pickle', mode='wb') as f:
            dump(result, f)
