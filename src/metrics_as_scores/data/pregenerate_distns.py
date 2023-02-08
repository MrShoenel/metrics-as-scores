import pandas as pd
from typing import Any
from nptyping import Float, NDArray, Shape
from tqdm import tqdm
from numpy.random import default_rng
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from joblib import Parallel, delayed
from metrics_as_scores.distribution.distribution import DistTransform, Dataset
from metrics_as_scores.data.pregenerate_fit import fit, Fitter
from sklearn.model_selection import ParameterGrid




def generate_parametric_fits(ds: Dataset, num_jobs: int, fitter_type: type[Fitter], dist_transform: DistTransform, selected_rvs_c: list[type[rv_continuous]], selected_rvs_d: list[type[rv_discrete]], data_dict: dict[str, NDArray[Shape["*"], Float]], transform_values_dict: dict[str, float], data_discrete_dict: dict[str, NDArray[Shape["*"], Float]], transform_values_discrete_dict: dict[str, float]) -> list[dict[str, Any]]:
    # The thinking is this: To each data series we can always fit a continuous distribution,
    # whether it's discrete or continuous data. The same is not true the other way round, i.e.,
    # we should not fit a discrete distribution if the data is known to be continuous.
    # Therefore, we do the following:
    #
    # - Regardless of the data, always attempt to fit a continuous RV
    # - For all discrete data, also attempt to fit a discrete RV
    #
    # That means that for discrete data, we will have two kinds of fitted RVs.
    # Also, when fitting a continuous RV to discrete data, we will add jitter to the data.

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

