from os import getcwd
from sys import path
path.append(getcwd())

import pandas as pd
import numpy as np
from pickle import dump
from joblib import Parallel, delayed
from scipy.integrate import quad, quadrature
from scipy.optimize import direct
from src.data.metrics import MetricID
from src.distribution.distribution import DensityFunc, DistTransform, Distribution, ECDF, KDECDF_approx, ParametricCDF
from sklearn.model_selection import ParameterGrid




def generate_densities(distr: Distribution, dens_fun: type[DensityFunc]=ECDF, unique_vals: bool=None, resample_samples=250_000, dist_transform: DistTransform=DistTransform.NONE) -> dict[str, DensityFunc]:
    metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
    metrics_ideal_df.replace({ np.nan: None }, inplace=True)
    metrics_ideal = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_ideal_df.Metric), metrics_ideal_df.Ideal) }

    systems_domains_df = pd.read_csv('./files/systems-domains.csv')
    systems_domains = dict(zip(systems_domains_df.System, systems_domains_df.Domain))
    systems_qc_names = dict(zip(systems_domains_df.System, systems_domains_df.System_QC_name))
    domains = systems_domains_df.Domain.unique().tolist()
    domains.append('__ALL__')

    param_grid = { 'domain': domains, 'metric': list(map(lambda m: m.name, MetricID)) }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))

    def get_cdf(grid_idx: int) -> tuple[str, DensityFunc]:
        row = expanded_grid.iloc[grid_idx,]
        domain = row.domain
        metric_id = MetricID[row.metric]

        if domain == '__ALL__':
            systems = list(systems_qc_names.values())
        else:
            # Gather all systems with the selected domain.
            temp = filter(lambda di: di[1] == domain, systems_domains.items())
            # Map the names to the Qualitas compiled corpus:
            systems = list(map(lambda di: systems_qc_names[di[0]], temp))
        
        # Only use unique values if explicitly requested or CDF-type is ECDF!
        uvals = True if unique_vals else (dens_fun == ECDF)
        data = distr.get_cdf_data(metric_id=metric_id, unique_vals=uvals, systems=systems)

        if dens_fun == ParametricCDF:
            try:
                df = Distribution.fit_parametric(data=data, max_samples=20_000, metric_id=metric_id, domain=domain, dist_transform=dist_transform)
                return (f'{domain}_{row.metric}', df)
            except Exception as e:
                print(f'Cannot fit parametric distribution for domain={domain}, metric={metric_id.value}')
                return (f'{domain}_{row.metric}', None)

        # Do transformation manually for other types of DensityFunc
        transform_value, data = Distribution.transform(data=data, dist_transform=dist_transform)

        return (f'{domain}_{row.metric}', dens_fun(data=data, resample_samples=resample_samples, compute_ranges=True, ideal_value=metrics_ideal[metric_id], dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain))

    cdfs = Parallel(n_jobs=-1)(delayed(get_cdf)(i) for i in range(len(expanded_grid.index)))
    return dict(cdfs)


if __name__ == '__main__':
    clazzes = [ParametricCDF] # [ECDF, KDECDF_approx, ParametricCDF]
    transfs = list(DistTransform)

    d = Distribution(df=pd.read_csv('csv/metrics.csv'))    
    for transf in transfs:
        for clazz in clazzes:
            temp = generate_densities(distr=d, dens_fun=clazz, unique_vals=True, resample_samples=75_000, dist_transform=transf)
            with open(f'./results/cdfs_{clazz.__name__}_{transf.name}.pickle', 'wb') as f:
                dump(temp, f)
            print(f'Finished generating CDFs for {clazz.__name__} with transform {transf.name}.')
