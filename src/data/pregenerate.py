from os import getcwd
from sys import path
from typing import Any, Union

from src.tools.funcs import flatten_dict
path.append(getcwd())

import pandas as pd
import numpy as np
from pickle import dump
from joblib import Parallel, delayed
from scipy.stats import norm
from src.data.metrics import Dataset, MetricID
from src.data.pregenerate_fit import Continuous_RVs_dict, Discrete_RVs_dict
from src.distribution.distribution import Density, DistTransform, Dataset, Empirical, KDE_approx, Parametric, Parametric_discrete
from sklearn.model_selection import ParameterGrid




def generate_densities(distr: Dataset, dens_fun: type[Density]=Empirical, unique_vals: bool=None, resample_samples=250_000, dist_transform: DistTransform=DistTransform.NONE) -> dict[str, Density]:
    metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
    metrics_ideal_df.replace({ np.nan: None }, inplace=True)
    metrics_ideal = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_ideal_df.Metric), metrics_ideal_df.Ideal) }

    domains = Dataset.domains()
    param_grid = { 'domain': domains, 'metric': list(map(lambda m: m.name, MetricID)) }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))

    def get_density(grid_idx: int) -> tuple[str, Density]:
        row = expanded_grid.iloc[grid_idx,]
        domain = row.domain
        systems = Dataset.systems_for_domain(domain=domain)
        metric_id = MetricID[row.metric]
        
        # Only use unique values if explicitly requested or CDF-type is ECDF!
        uvals = True if unique_vals else (dens_fun == Empirical)
        data = distr.data(metric_id=metric_id, unique_vals=uvals, systems=systems)

        if dens_fun == ParametricCDF:
            df = None
            try:
                df = Distribution.fit_parametric(data=data, alpha=0.001, max_samples=30_000, metric_id=metric_id, domain=domain, dist_transform=dist_transform)
            except Exception as e:
                df = ParametricCDF(dist=norm, pval=np.nan, dstat=np.nan, dist_params=None, range=(np.min(data), np.max(data)), compute_ranges=False, ideal_value=np.nan, dist_transform=dist_transform, transform_value=np.nan, metric_id=metric_id, domain=domain)
            return (f'{domain}_{row.metric}', df)

        # Do transformation manually for other types of DensityFunc
        transform_value, data = Dataset.transform(data=data, dist_transform=dist_transform)

        return (f'{domain}_{row.metric}', dens_fun(data=data, resample_samples=resample_samples, compute_ranges=True, ideal_value=metrics_ideal[metric_id], dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain))

    cdfs = Parallel(n_jobs=-1)(delayed(get_density)(i) for i in range(len(expanded_grid.index)))
    return dict(cdfs)


def fits_to_MaS_densities(df: pd.DataFrame, dist_transform: DistTransform, use_continuous: bool) -> dict[str, Union[Parametric, Parametric_discrete]]:
    data_df = pd.read_csv('./csv/metrics.csv')

    metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
    metrics_ideal_df.replace({ np.nan: None }, inplace=True)
    metrics_ideal = { x: y for (x, y) in zip(metrics_ideal_df.Metric, metrics_ideal_df.Ideal) }

    domains = Dataset.domains(include_all_domain=True)
    metrics = list(MetricID)
    the_type = 'continuous' if use_continuous else 'discrete'
    use_stat = 'stat_tests_tests_ks_2samp_ordinary_stat' if use_continuous else 'stat_tests_tests_epps_singleton_2samp_jittered_stat'
    use_pval = 'stat_tests_tests_ks_2samp_ordinary_pval' if use_continuous else 'stat_tests_tests_epps_singleton_2samp_jittered_pval'
    use_vars = Continuous_RVs_dict if use_continuous else Discrete_RVs_dict
    Use_class = Parametric if use_continuous else Parametric_discrete

    the_dict: dict[str, Parametric] = {}
    for domain in domains:
        for metric in metrics:
            key = f'{domain}_{metric.name}'
            candidates = df[(df.domain == domain) & (df.metric == metric.name) & (df.type == the_type) & (df.dist_transform == dist_transform.value)]
            if len(candidates.index) == 0:
                # No fit at all :(
                the_dict[key] = Use_class.unfitted(dist_transform=dist_transform)
            else:
                candidates.sort_values(by=[use_stat], ascending=True, inplace=True) # Lowest D-stat first
                best = candidates.head(1).iloc[0,]
                dist_type = use_vars[best.rv]
                dist = dist_type()
                params = ()
                for pi in dist._param_info():
                    params += (best[f'params_{pi.name}'],)
                
                data = data_df[(data_df.metric == metric.name)]
                if domain != '__ALL__':
                    data = data[(data.domain == domain)]
                data = data.value.to_numpy()
                
                the_dict[key] = Use_class(dist=dist, pval=best[use_pval], dstat=best[use_stat], dist_params=params, range=(data.min(), data.max()),
                    compute_ranges=True, ideal_value=metrics_ideal[best.metric], dist_transform=dist_transform,
                    transform_value=best.transform_value, metric_id=metric, domain=domain)
    
    return the_dict


if __name__ == '__main__':
    clazzes = [Empirical, KDE_approx] # We'll do Parametric below.
    transfs = list(DistTransform)

    d = Dataset(df=pd.read_csv('csv/metrics.csv'))
    for transf in transfs:
        for clazz in clazzes:
            temp = generate_densities(distr=d, dens_fun=clazz, unique_vals=True, resample_samples=75_000, dist_transform=transf)
            with open(f'./results/densities_{clazz.__name__}_{transf.name}.pickle', 'wb') as f:
                dump(temp, f)
            print(f'Finished generating Densities for {clazz.__name__} with transform {transf.name}.')
                dump(temp, f)
            print(f'Finished generating CDFs for {clazz.__name__} with transform {transf.name}.')
