from pathlib import Path
import pandas as pd
from typing import Any, Union
from warnings import warn
from os.path import exists
from pickle import dump, load
from joblib import Parallel, delayed
from tqdm import tqdm
from metrics_as_scores.tools.funcs import flatten_dict
from metrics_as_scores.data.pregenerate_fit import Continuous_RVs_dict, Discrete_RVs_dict
from metrics_as_scores.distribution.distribution import Dataset, Density, DistTransform, Dataset, Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete
from metrics_as_scores.distribution.fitting import StatisticalTest
from sklearn.model_selection import ParameterGrid




def generate_densities(dataset: Dataset, clazz: type[Density]=Empirical, unique_vals: bool=None, resample_samples=250_000, dist_transform: DistTransform=DistTransform.NONE) -> dict[str, Density]:
    contexts = list(dataset.contexts(include_all_contexts=True))
    param_grid = { 'context': contexts, 'qtype': dataset.quantity_types }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))

    def get_density(grid_idx: int) -> tuple[str, Density]:
        row = expanded_grid.iloc[grid_idx,]
        context = row.context
        qtype = row.qtype
        
        # Only use unique values if explicitly requested or CDF-type is ECDF!
        uvals = True if unique_vals else (clazz == Empirical)
        data = dataset.data(qtype=qtype, context=None if context == '__ALL__' else context, unique_vals=uvals)

        # Do transformation manually for other types of DensityFunc
        transform_value, data = Dataset.transform(data=data, dist_transform=dist_transform)

        return (f'{context}_{qtype}', clazz(data=data, resample_samples=resample_samples, compute_ranges=True, ideal_value=dataset.ideal_values[qtype], dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context))

    cdfs = Parallel(n_jobs=-1)(delayed(get_density)(i) for i in tqdm(range(len(expanded_grid.index))))
    return dict(cdfs)


def fits_to_MAS_densities(dataset: Dataset, distns_dict: dict[int, dict[str, Any]], dist_transform: DistTransform, use_continuous: bool) -> dict[str, Union[Parametric, Parametric_discrete]]:
    df = pd.DataFrame([flatten_dict(d) for d in distns_dict.values()])
    data_df = dataset.df

    contexts = list(dataset.contexts(include_all_contexts=True))
    the_type = 'continuous' if use_continuous else 'discrete'
    use_test = 'ks_2samp_ordinary' if use_continuous else 'epps_singleton_2samp_jittered'
    use_stat = f'stat_tests_tests_{use_test}_stat'
    use_vars = Continuous_RVs_dict if use_continuous else Discrete_RVs_dict
    Use_class = Parametric if use_continuous else Parametric_discrete

    the_dict: dict[str, Parametric] = {}
    for context in contexts:
        for qtype in dataset.quantity_types:
            key = f'{context}_{qtype}'
            candidates = df[(df.context == context) & (df.qtype == qtype) & (df.type == the_type) & (df.dist_transform == dist_transform.name)]
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
                
                data = data_df[(data_df[dataset.ds['colname_type']] == qtype)]
                if context != '__ALL__':
                    data = data[(data[dataset.ds['colname_context']] == context)]
                data = data[dataset.ds['colname_data']].to_numpy()
                
                the_dict[key] = Use_class(dist=dist, stat_tests=stat_tests_dict, use_stat_test=use_test, dist_params=params, range=(data.min(), data.max()),
                    compute_ranges=True, ideal_value=dataset.ideal_values[best.qtype], dist_transform=dist_transform,
                    transform_value=best.transform_value, qtype=qtype, context=context)
    
    return the_dict



def generate_empirical(dataset: Dataset, densities_dir: Path, clazz: Union[Empirical, KDE_approx], transform: DistTransform):
    temp = generate_densities(dataset=dataset, clazz=clazz, dist_transform=transform, unique_vals=True, resample_samples=75_000)
    dens_file = str(densities_dir.joinpath(f'./densities_{clazz.__name__}_{transform.name}.pickle'))
    with open(file=dens_file, mode='wb') as fp:
        dump(obj=temp, file=fp)
    print(f'Finished generating Densities for {clazz.__name__} with transform {transform.name}.')



def generate_parametric(dataset: Dataset, densities_dir: Path, fits_dir: Path, clazz: Union[Parametric, Parametric_discrete], transform: DistTransform):
    fits_file = str(fits_dir.joinpath(f'./pregen_distns_{transform.name}.pickle').resolve())
    if not exists(fits_file):
        warn(f'Cannot generate parametric distribution for {clazz.__name__} and transformation {transform.name}, because the file {fits_file} does not exist. Did you forget to create the fits using the script pregenerate_distns.py?')
        return

    with open(fits_file, 'rb') as f:
        distns_list = load(f)
        distns_dict = { item['grid_idx']: item for item in distns_list }
    use_continuous = clazz == Parametric
    temp = fits_to_MAS_densities(dataset=dataset, distns_dict=distns_dict, dist_transform=transform, use_continuous=use_continuous)
    dens_file = str(densities_dir.joinpath(f'./densities_{clazz.__name__}_{transform.name}.pickle'))
    with open(file=dens_file, mode='wb') as f:
        dump(temp, f)
    print(f'Finished generating parametric Densities for {clazz.__name__} with transform {transform.name}.')



def generate_empirical_discrete(dataset: Dataset, densities_dir: Path, transform: DistTransform):
    the_dict: dict[str, Empirical_discrete] = {}

    for context in dataset.contexts(include_all_contexts=True):
        use_context = None if context == '__ALL__' else context
        for qtype in dataset.quantity_types:
            key = f'{context}_{qtype}'

            if not dataset.is_qtype_discrete(qtype=qtype):
                the_dict[key] = Empirical_discrete.unfitted(dist_transform=transform)
            else:
                data = dataset.data(qtype=qtype, context=use_context, unique_vals=False)
                transform_value, data = Dataset.transform(data=data, dist_transform=transform, continuous_value=False)
                the_dict[key] = Empirical_discrete(data=data.astype(int), ideal_value=dataset.ideal_values[qtype], dist_transform=transform, transform_value=transform_value, qtype=qtype, context=context)

    dens_file = str(densities_dir.joinpath(f'./densities_{Empirical_discrete.__name__}_{transform.name}.pickle'))
    with open(file=dens_file, mode='wb') as fp:
        dump(obj=the_dict, file=fp)
    print(f'Finished generating empirical Densities for {Empirical_discrete.__name__} with transform {transform.name}.')





