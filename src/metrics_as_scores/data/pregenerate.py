from os import cpu_count
from sys import path
from pathlib import Path

this_dir = Path(__file__).resolve().parent
src_dir = this_dir.parent.parent
path.append(str(src_dir.resolve()))



import pandas as pd
import numpy as np
from typing import Any, Union
from warnings import warn
from os.path import exists
from pickle import dump, load
from joblib import Parallel, delayed
from metrics_as_scores.tools.funcs import flatten_dict
from metrics_as_scores.data.metrics import MetricID
from metrics_as_scores.data.pregenerate_fit import Continuous_RVs_dict, Discrete_RVs_dict
from metrics_as_scores.distribution.distribution import Dataset, Density, DistTransform, Dataset, Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete
from metrics_as_scores.distribution.fitting import StatisticalTest
from sklearn.model_selection import ParameterGrid




def generate_densities(dataset: Dataset, clazz: type[Density]=Empirical, unique_vals: bool=None, resample_samples=250_000, dist_transform: DistTransform=DistTransform.NONE) -> dict[str, Density]:
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
        uvals = True if unique_vals else (clazz == Empirical)
        data = dataset.data(metric_id=metric_id, unique_vals=uvals, systems=systems)

        # Do transformation manually for other types of DensityFunc
        transform_value, data = Dataset.transform(data=data, dist_transform=dist_transform)

        return (f'{domain}_{row.metric}', clazz(data=data, resample_samples=resample_samples, compute_ranges=True, ideal_value=metrics_ideal[metric_id], dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain))

    cdfs = Parallel(n_jobs=-1)(delayed(get_density)(i) for i in range(len(expanded_grid.index)))
    return dict(cdfs)


def fits_to_MaS_densities(dataset: Dataset, distns_dict: dict[int, dict[str, Any]], dist_transform: DistTransform, use_continuous: bool) -> dict[str, Union[Parametric, Parametric_discrete]]:
    df = pd.DataFrame([flatten_dict(d) for d in distns_dict.values()])
    data_df = dataset.df
    metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
    metrics_ideal_df.replace({ np.nan: None }, inplace=True)
    metrics_ideal = { x: y for (x, y) in zip(metrics_ideal_df.Metric, metrics_ideal_df.Ideal) }

    domains = Dataset.domains(include_all_domain=True)
    metrics = list(MetricID)
    the_type = 'continuous' if use_continuous else 'discrete'
    use_test = 'ks_2samp_ordinary' if use_continuous else 'epps_singleton_2samp_jittered'
    use_stat = f'stat_tests_tests_{use_test}_stat'
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
                candidates = candidates.sort_values(by=[use_stat], ascending=True, inplace=False) # Lowest D-stat first
                best = candidates.head(1).iloc[0,]
                stat_tests_dict = StatisticalTest.from_dict(d=best, key_prefix='stat_tests_tests_')
                dist_type = use_vars[best.rv]
                dist = dist_type()
                params = ()
                for pi in dist._param_info():
                    params += (best[f'params_{pi.name}'],)
                
                data = data_df[(data_df.Metric == metric.name)]
                if domain != '__ALL__':
                    data = data[(data.Domain == domain)]
                data = data.Value.to_numpy()
                
                the_dict[key] = Use_class(dist=dist, stat_tests=stat_tests_dict, use_stat_test=use_test, dist_params=params, range=(data.min(), data.max()),
                    compute_ranges=True, ideal_value=metrics_ideal[best.metric], dist_transform=dist_transform,
                    transform_value=best.transform_value, metric_id=metric, domain=domain)
    
    return the_dict



def generate_empirical(dataset: Dataset, clazz: Union[Empirical, KDE_approx], transform: DistTransform):
    temp = generate_densities(dataset=dataset, clazz=clazz, dist_transform=transform, unique_vals=True, resample_samples=75_000)
    with open(f'./results/densities_{clazz.__name__}_{transform.name}.pickle', 'wb') as f:
        dump(temp, f)
    print(f'Finished generating Densities for {clazz.__name__} with transform {transform.name}.')



def generate_parametric(dataset: Dataset, clazz: Union[Parametric, Parametric_discrete], transform: DistTransform):
    use_file = f'./results/pregnerate_distns_{transform.name}.pickle'
    if not exists(use_file):
        warn(f'Cannot generate parametric distribution for {clazz.__name__} and transformation {transform.name}, because the file {use_file} does not exist. Did you forget to create the fits using the script pregenerate_distns.py?')
        return

    with open(use_file, 'rb') as f:
        distns_list = load(f)
        distns_dict = { item['grid_idx']: item for item in distns_list }
    use_continuous = clazz == Parametric
    temp = fits_to_MaS_densities(dataset=dataset, distns_dict=distns_dict, dist_transform=transform, use_continuous=use_continuous)
    with open(f'./results/densities_{clazz.__name__}_{transform.name}.pickle', 'wb') as f:
        dump(temp, f)
    print(f'Finished generating parametric Densities for {clazz.__name__} with transform {transform.name}.')



def generate_empirical_discrete(dataset: Dataset, transform: DistTransform):
    metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
    metrics_ideal_df.replace({ np.nan: None }, inplace=True)
    metrics_ideal = { x: y for (x, y) in zip(metrics_ideal_df.Metric, metrics_ideal_df.Ideal) }

    metrics_discrete_df = pd.read_csv('./files/metrics-discrete.csv')
    metrics_discrete = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_discrete_df.Metric), metrics_discrete_df.Discrete) }
    the_dict: dict[str, Empirical_discrete] = {}

    for domain in Dataset.domains(include_all_domain=True):
        use_domain = None if domain == '__ALL__' else domain
        for metric_id in list(MetricID):
            key = f'{domain}_{metric_id.name}'

            if not metrics_discrete[metric_id]:
                the_dict[key] = Empirical_discrete.unfitted(dist_transform=transform)
            else:
                data = dataset.data(metric_id=metric_id, domain=use_domain, unique_vals=False)
                transform_value, data = Dataset.transform(data=data, dist_transform=transform, continuous_value=False)
                the_dict[key] = Empirical_discrete(data=data.astype(int), ideal_value=metrics_ideal[metric_id.name], dist_transform=transform, transform_value=transform_value, metric_id=metric_id, domain=use_domain)

    with open(f'./results/densities_{Empirical_discrete.__name__}_{transform.name}.pickle', 'wb') as f:
        dump(the_dict, f)
    print(f'Finished generating empirical Densities for {Empirical_discrete.__name__} with transform {transform.name}.')






if __name__ == '__main__':
    dataset = Dataset(df=pd.read_csv('csv/metrics.csv'))

    grid = dict(
        clazz = [Parametric, Parametric_discrete],
        transform = list(DistTransform))
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=grid))
    Parallel(n_jobs=min(cpu_count(), len(expanded_grid.index)))(delayed(generate_parametric)(dataset, expanded_grid.iloc[i,]['clazz'], expanded_grid.iloc[i,]['transform']) for i in range(len(expanded_grid.index)))


    grid = dict(
        clazz = [Empirical, KDE_approx],
        transform = list(DistTransform))
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=grid))
    Parallel(n_jobs=min(cpu_count(), len(expanded_grid.index)))(delayed(generate_empirical)(dataset, expanded_grid.iloc[i,]['clazz'], expanded_grid.iloc[i,]['transform']) for i in range(len(expanded_grid.index)))

    # Also, let's pre-generate the discrete empiricals.
    grid = dict(
        clazz = [Empirical_discrete],
        transform = list(DistTransform))
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=grid))
    Parallel(n_jobs=min(cpu_count(), len(expanded_grid.index)))(delayed(generate_empirical_discrete)(dataset, expanded_grid.iloc[i,]['transform']) for i in range(len(expanded_grid.index)))
