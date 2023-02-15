from pathlib import Path
import pandas as pd
from typing import Union
from warnings import warn
from os.path import exists
from pickle import dump, load
from joblib import Parallel, delayed
from tqdm import tqdm
from metrics_as_scores.tools.funcs import flatten_dict
from metrics_as_scores.data.pregenerate_fit import Continuous_RVs_dict, Discrete_RVs_dict, FitResult
from metrics_as_scores.distribution.distribution import Dataset, Density, DistTransform, Dataset, Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete
from metrics_as_scores.distribution.fitting import StatisticalTest
from sklearn.model_selection import ParameterGrid




def generate_densities(
    dataset: Dataset,
    clazz: type[Density]=Empirical,
    unique_vals: bool=None,
    resample_samples=250_000,
    dist_transform: DistTransform=DistTransform.NONE
) -> dict[str, Density]:
    """
    Generates a set of :py:class:`Density` objects for a certain :py:class:`DistTransform`.
    For each combination, we will later save one file that is then to be used in the web
    application, as generating these on-the-fly would take too long.

    dataset: ``Dataset``
        Required for obtaining quantity types, contexts, and filtered data.
    
    clazz: ``type[Density]``
        A type of empirical density to generate densities for.
    
    unique_vals: ``bool``
        Used to conditionally add some jitter to data to all data points unique. This is
        automatically set to `True` if the class is :py:class:`Empirical`, because this
        class is for continuous RVs. If the data is not continuous (real), then setting
        this to `True` will make it so.
    
    resample_samples: ``int``
        Unsigned integer, passed forward to the type of dict[str, Density].
    
    dist_transform: ``DistTransform``
        The chosen transformation for the data.
    
    :rtype: ``dict[str, Density]``

    :return:
        A dictionary where the key is made of the context and quantity type, and
        the value is the generated :py:class:`Empirical` density.
    """
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


def fits_to_MAS_densities(
    dataset: Dataset,
    distns_dict: dict[int, FitResult],
    dist_transform: DistTransform,
    use_continuous: bool
) -> dict[str, Union[Parametric, Parametric_discrete]]:
    """
    Converts previously produced parametric fits to :py:class:`Density` objects that can
    be loaded and used in the web application.
    Similar to :py:meth:`generate_densities()`, this method also returns a dictionary
    with generated parametric densities.

    dataset: ``Dataset``
        Required for obtaining quantity types, contexts, and filtered data.
    
    distns_dict: ``dict[int, FitResult]``
        Dictionary with all fit results for a data transform. The `int`-key is just the
        previously used grid index and not relevant here.
    
    dist_transform: ``DistTransform``
        The chosen transformation for the data.
    
    use_continuous: ``bool``
        Used to select and generate densities based on either continuous (`True`) RVs or
        discrete RVs.
    
    :rtype: ``dict[str, Union[Parametric, Parametric_discrete]]``

    :return:
        A dictionary where the key is made of the context and quantity type, and
        the value is the generated :py:class:`Union[Parametric, Parametric_discrete]` density.
    """
    df = pd.DataFrame([flatten_dict(d) for d in distns_dict.values()])
    data_df = dataset.df

    contexts = list(dataset.contexts(include_all_contexts=True))
    the_type = 'continuous' if use_continuous else 'discrete'
    use_test = 'ks_2samp_ordinary' if use_continuous else 'epps_singleton_2samp_jittered'
    use_stat = f'stat_tests_tests_{use_test}_stat'
    use_vars = Continuous_RVs_dict if use_continuous else Discrete_RVs_dict
    Use_class = Parametric if use_continuous else Parametric_discrete

    df_cols = list(df.columns)
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
                    use_key = f'params_{pi.name}'
                    if pi.name.endswith('_') and not use_key in df_cols:
                        use_key = use_key.rstrip('_')
                    try:
                        params += (best[use_key],)
                    except KeyError:
                        # Happens when the candidate was not actually fit, e.g., when a discrete RV
                        # was selected for continuous data.
                        the_dict[key] = Use_class.unfitted(dist_transform=dist_transform)
                        continue
                
                data = data_df[(data_df[dataset.ds['colname_type']] == qtype)]
                if context != '__ALL__':
                    data = data[(data[dataset.ds['colname_context']] == context)]
                data = data[dataset.ds['colname_data']].to_numpy()
                
                the_dict[key] = Use_class(dist=dist, stat_tests=stat_tests_dict, use_stat_test=use_test, dist_params=params, range=(data.min(), data.max()),
                    compute_ranges=True, ideal_value=dataset.ideal_values[best.qtype], dist_transform=dist_transform,
                    transform_value=best.transform_value, qtype=qtype, context=context)
    
    return the_dict



def generate_empirical(
    dataset: Dataset,
    densities_dir: Path,
    clazz: Union[Empirical, KDE_approx],
    transform: DistTransform
) -> None:
    """
    Generates a set of empirical (continuous) densities for a given density type
    (Empirical or KDE_Approx) and data transform.

    dataset: ``Dataset``
        Required for obtaining quantity types, contexts, and filtered data.
    
    densities_dir: ``Path``
        The directory to store the generated densities. The resulting file is a key
        of the used density type and data transform.
    
    clazz: ``Union[Empirical, KDE_approx]``
        The type of density you wish to create.
    
    transform: ``DistTransform``
        The chosen transformation for the data.
    
    :return:
        This method does not return anything but only writes the result to disk.
    """
    temp = generate_densities(dataset=dataset, clazz=clazz, dist_transform=transform, unique_vals=True, resample_samples=75_000)
    dens_file = str(densities_dir.joinpath(f'./densities_{clazz.__name__}_{transform.name}.pickle'))
    with open(file=dens_file, mode='wb') as fp:
        dump(obj=temp, file=fp)
    print(f'Finished generating Densities for {clazz.__name__} with transform {transform.name}.')



def generate_parametric(
    dataset: Dataset,
    densities_dir: Path,
    fits_dir: Path,
    clazz: Union[Parametric, Parametric_discrete],
    transform: DistTransform
) -> None:
    """
    Generates a set of parametric densities for a given density type
    (Parametric or Parametric_discrete) and data transform.

    dataset: ``Dataset``
        Required for obtaining quantity types, contexts, and filtered data.
    
    densities_dir: ``Path``
        The directory to store the generated densities. The resulting file is a key
        of the used density type and data transform.
    
    clazz: ``Union[Parametric, Parametric_discrete]``
        The type of density you wish to create.
    
    transform: ``DistTransform``
        The chosen transformation for the data.
    
    :return:
        This method does not return anything but only writes the result to disk.
    """
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



def generate_empirical_discrete(
    dataset: Dataset,
    densities_dir: Path,
    transform: DistTransform
) -> None:
    """
    Generates discrete empirical densities for a given data transform. Only uses
    the type :py:class:``Empirical_discrete`` for this.

    dataset: ``Dataset``
        Required for obtaining quantity types, contexts, and filtered data.
    
    densities_dir: ``Path``
        The directory to store the generated densities. The resulting file is a key
        of the used density type and data transform.
    
    transform: ``DistTransform``
        The chosen transformation for the data.
    
    :return:
        This method does not return anything but only writes the result to disk.
    """
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
