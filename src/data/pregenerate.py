from os import getcwd
from sys import path
path.append(getcwd())

import pandas as pd
import numpy as np
from pickle import dump
from joblib import Parallel, delayed
from scipy.integrate import quad, quadrature
from scipy.optimize import direct
from scipy.stats import norm
from src.data.metrics import MetricID
from src.distribution.distribution import DensityFunc, DistTransform, Distribution, ECDF, KDECDF_approx, ParametricCDF
from sklearn.model_selection import ParameterGrid




def generate_densities(distr: Distribution, dens_fun: type[DensityFunc]=ECDF, unique_vals: bool=None, resample_samples=250_000, dist_transform: DistTransform=DistTransform.NONE) -> dict[str, DensityFunc]:
    metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
    metrics_ideal_df.replace({ np.nan: None }, inplace=True)
    metrics_ideal = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_ideal_df.Metric), metrics_ideal_df.Ideal) }

    domains = Distribution.domains()
    param_grid = { 'domain': domains, 'metric': list(map(lambda m: m.name, MetricID)) }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))

    def get_cdf(grid_idx: int) -> tuple[str, DensityFunc]:
        row = expanded_grid.iloc[grid_idx,]
        domain = row.domain
        systems = Distribution.systems_for_domain(domain=domain)
        metric_id = MetricID[row.metric]
        
        # Only use unique values if explicitly requested or CDF-type is ECDF!
        uvals = True if unique_vals else (dens_fun == ECDF)
        data = distr.data(metric_id=metric_id, unique_vals=uvals, systems=systems)

        if dens_fun == ParametricCDF:
            df = None
            try:
                df = Distribution.fit_parametric(data=data, alpha=0.001, max_samples=30_000, metric_id=metric_id, domain=domain, dist_transform=dist_transform)
            except Exception as e:
                df = ParametricCDF(dist=norm, pval=np.nan, dstat=np.nan, dist_params=None, range=(np.min(data), np.max(data)), compute_ranges=False, ideal_value=np.nan, dist_transform=dist_transform, transform_value=np.nan, metric_id=metric_id, domain=domain)
            return (f'{domain}_{row.metric}', df)

        # Do transformation manually for other types of DensityFunc
        transform_value, data = Distribution.transform(data=data, dist_transform=dist_transform)

        return (f'{domain}_{row.metric}', dens_fun(data=data, resample_samples=resample_samples, compute_ranges=True, ideal_value=metrics_ideal[metric_id], dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain))

    cdfs = Parallel(n_jobs=-1)(delayed(get_cdf)(i) for i in range(len(expanded_grid.index)))
    return dict(cdfs)


if __name__ == '__main__':
    clazzes = [ECDF, KDECDF_approx, ParametricCDF]
    transfs = list(DistTransform)

    d = Distribution(df=pd.read_csv('csv/metrics.csv'))    
    for transf in transfs:
        for clazz in clazzes:
            temp = generate_densities(distr=d, dens_fun=clazz, unique_vals=True, resample_samples=75_000, dist_transform=transf)
            with open(f'./results/cdfs_{clazz.__name__}_{transf.name}.pickle', 'wb') as f:
                dump(temp, f)
            print(f'Finished generating CDFs for {clazz.__name__} with transform {transf.name}.')
