from os import getcwd
from sys import path
path.append(getcwd())

import pandas as pd
from pickle import dump
from joblib import Parallel, delayed
from src.data.metrics import MetricID
from src.distribution.distribution import DensityFunc, Distribution, ECDF, KDECDF_approx
from sklearn.model_selection import ParameterGrid


def generate_densities(distr: Distribution, CDF: type[DensityFunc]=ECDF, unique_vals: bool=None, resample_samples=250_000) -> dict[str, DensityFunc]:
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
        uvals = True if unique_vals else (CDF == ECDF)
        data = distr.get_cdf_data(metric_id=metric_id, unique_vals=uvals, systems=systems)
        return (f'{domain}_{row.metric}', CDF(data=data, resample_samples=resample_samples, compute_ranges=True))

    cdfs = Parallel(n_jobs=-1)(delayed(get_cdf)(i) for i in range(len(expanded_grid.index)))
    return dict(cdfs)


if __name__ == '__main__':
    d = Distribution(df=pd.read_csv('csv/metrics.csv'))    
    temp1 = generate_densities(distr=d, CDF=KDECDF_approx, unique_vals=True, resample_samples=75_000)
    with open('./results/cdfs_KDECDF_approx.pickle', 'wb') as f:
        dump(temp1, f)

    temp = generate_densities(distr=d, CDF=ECDF)
    with open('./results/cdfs_ECDF.pickle', 'wb') as f:
        dump(temp, f)
