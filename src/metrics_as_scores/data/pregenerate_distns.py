import pandas as pd
from typing import Any
from nptyping import Float, NDArray, Shape
from tqdm import tqdm
from numpy.random import default_rng
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from joblib import Parallel, delayed
from metrics_as_scores.distribution.distribution import DistTransform, Dataset
from metrics_as_scores.data.pregenerate_fit import fit, Fitter, FitResult
from sklearn.model_selection import ParameterGrid




def generate_parametric_fits(
    ds: Dataset,
    num_jobs: int,
    fitter_type: type[Fitter],
    dist_transform: DistTransform,
    selected_rvs_c: list[type[rv_continuous]],
    selected_rvs_d: list[type[rv_discrete]],
    data_dict: dict[str, NDArray[Shape["*"], Float]],
    transform_values_dict: dict[str, float],
    data_discrete_dict: dict[str, NDArray[Shape["*"], Float]],
    transform_values_discrete_dict: dict[str, float]
) -> list[FitResult]:
    """
    The thinking is this: To each data series we can always fit a continuous distribution,
    whether it's discrete or continuous data. The same is not true the other way round, i.e.,
    we must not fit a discrete distribution if the data is known to be continuous.
    Therefore, we do the following:
    
    - Regardless of the data, always attempt to fit a continuous RV
    - For all discrete data, also attempt to fit a discrete RV
    
    That means that for discrete data, we will have two kinds of fitted RVs.
    Also, when fitting a continuous RV to discrete data, we will add jitter to the data.


    ds: ``Dataset``
        The data, needed for obtaining quantity types and contexts. Also passed forward to
        :py:meth:`fit()`.
    
    num_jobs: ``int``
        Degree of parallelization used.
    
    fitter_type: ``type[Fitter]``
        The class for the fitter to use, either :py:class:`Fitter` or :py:class:`FitterPymoo`.
    
    dist_transform: ``DistTransform``
        The transform for which to generate parametric fits for. Later, we will save a single
        file per transform, containing all related fits.
    
    selected_rvs_c: ``list[type[rv_continuous]]``
        Continuous RVs to attempt to fit.
    
    selected_rvs_d: ``list[type[rv_discrete]]``
        Discrete RVs to attempt to fit.
    
    data_dict: ``dict[str, NDArray[Shape["*"], Float]]``
        A dictionary where they key consists of the context and the quantity type.
        For each entry, it contains a 1-D array of data used for fitting.
    
    transform_values_dict: ``dict[str, float]``
        Similar to ``data_dict``, this dictionary contains the transformation value that
        was used to transform the data in the 1-D array.
    
    data_discrete_dict: ``dict[str, NDArray[Shape["*"], Float]]``
        Like ``data_dict``, but for discrete RVs fitted to discrete data.
        
    transform_values_discrete_dict: ``dict[str, float]``
        Like ``transform_values_dict``, but for the discrete datas.
    
    :return:
        A list of :py:class:``FitResult`` objects.
    """

    param_grid = {
        'context': list(ds.contexts(include_all_contexts=True)),
        'qtype': ds.quantity_types, # Fit continuous to all
        'rv': list([rv.__name__ for rv in selected_rvs_c]), # continuous here, discrete below
        'type': ['continuous'],
        'dist_transform': [dist_transform] }
    expanded_grid = pd.DataFrame(ParameterGrid(param_grid=param_grid))

    discrete_types = ds.quantity_types_discrete
    if len(discrete_types) > 0:
        # Only fit discrete if we have it
        param_grid = {
            'context': list(ds.contexts(include_all_contexts=True)),
            'qtype': discrete_types,
            'rv': list([rv.__name__ for rv in selected_rvs_d]),
            'type': ['discrete'],
            'dist_transform': [dist_transform] }
        expanded_grid = pd.concat([
            expanded_grid, pd.DataFrame(ParameterGrid(param_grid=param_grid))])

    def get_datas(grid_idx) -> tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float], float]:
        row = expanded_grid.iloc[grid_idx,]
        key = f'{row.context}_{row.qtype}'
        key_u = f'{key}_u'
        discrete_discrete = ds.is_qtype_discrete(qtype=row.qtype) and row.rv in selected_rvs_d
        if discrete_discrete:
            # Fit a discrete RV to discrete data
            return (data_discrete_dict[key], data_discrete_dict[key_u], transform_values_discrete_dict[key])
        return (data_dict[key], data_dict[key_u], transform_values_dict[key])

    rng = default_rng(seed=76543210)
    indexes = rng.choice(a=list(range(len(expanded_grid))), replace=False, size=len(expanded_grid))
    return Parallel(n_jobs=num_jobs)(delayed(fit)(ds, fitter_type, i, expanded_grid.iloc[i,], dist_transform, *get_datas(grid_idx=i)) for i in tqdm(indexes))

