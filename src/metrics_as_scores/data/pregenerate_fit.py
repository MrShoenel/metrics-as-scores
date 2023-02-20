"""
This is an extra module that holds functions globally, such that we
can exploit multiprocessing effortlessly.
"""

import numpy as np
from pickle import dump
from typing import Any, TypedDict, Union
from nptyping import Float, NDArray, Shape
from metrics_as_scores.distribution.distribution import DistTransform, Dataset
from metrics_as_scores.distribution.fitting import Continuous_RVs, Discrete_RVs, Fitter, StatisticalTest, StatisticalTestJson
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from timeit import default_timer as timer



Continuous_RVs_dict: dict[str, type[rv_continuous]] = { x: y for (x, y) in zip(map(lambda rv: type(rv).__name__, Continuous_RVs), map(lambda rv: type(rv), Continuous_RVs)) }
"""
Dictionary of continuous random variables that are supported by `scipy.stats`.
Note this is a dictionary of types, rather than instances.
"""
Discrete_RVs_dict: dict[str, type[rv_discrete]] = { x: y for (x, y) in zip(map(lambda rv: type(rv).__name__, Discrete_RVs), map(lambda rv: type(rv), Discrete_RVs)) }
"""
Dictionary of discrete random variables that are supported by `scipy.stats`.
Note this is a dictionary of types, rather than instances.
"""



def get_data_tuple(
    ds: Dataset,
    qtype: str,
    dist_transform: DistTransform,
    continuous_transform: bool=True
) -> list[tuple[str, NDArray[Shape["*"], Float]]]:
    """
    This method is part of the workflow for computing parametric fits.
    For a specific type of quantity and transform, it creates datasets
    for all available contexts.

    ds: ``Dataset``

    qtype: ``str``
        The type of quantity to get datasets for.
    
    dist_transform: ``DistTransform``
        The chosen distribution transform.

    continuous_transform: ``bool``
        Whether the transform is real-valued or must be converted to integer.


    :rtype: ``list[tuple[str, NDArray[Shape["*"], Float]]]``

    :return: A list of tuples of three elements. The first element is a key
        that identifies the context, the quantity type, and whether the data
        was computed using unique values (see :meth:`Dataset.transform()`).
    """
    l = []
    for ctx in ds.contexts(include_all_contexts=True):
        for unique_vals in [True, False]:
            data = ds.data(qtype=qtype, context=(None if ctx == '__ALL__' else ctx), unique_vals=unique_vals, sub_sample=25_000)
            transform_value, data = Dataset.transform(data=data, dist_transform=dist_transform, continuous_value=continuous_transform)
            key = f"{ctx}_{qtype}{('_u' if unique_vals else '')}"
            l.append((key, data, transform_value))
    return l


class FitResult(TypedDict):
    """
    This class is derived from :py:class:`TypedDict` and holds all properties
    related to a single fit result, that is, a single specific configuration
    that was fit to a 1-D array of data.
    """
    grid_idx: int
    dist_transform: str
    transform_value: Union[float, None]
    params: dict[str, Union[float, int]]

    # Also, from row.to_dict():
    context: str
    qtype: str
    rv: str
    type: str

    stat_tests: StatisticalTestJson


def fit(
    ds: Dataset,
    fitter_type: type[Fitter],
    grid_idx: int, row,
    dist_transform: DistTransform,
    the_data: NDArray[Shape["*"], Float],
    the_data_unique: NDArray[Shape["*"], Float],
    transform_value: Union[float, None],
    write_temporary_results: bool=False
) -> FitResult:
    """
    This is the main stand-alone function that computes a single parametric fit to
    a single 1-D array of data. This function is used in Parallel contexts and,
    therefore, lives on module top level so it can be serialized.

    ds: ``Dataset``
        The data, needed for obtaining quantity types and contexts. Also passed forward to
        :py:meth:`fit()`.
    
    fitter_type: ``type[Fitter]``
        The class for the fitter to use, either :py:class:`Fitter` or :py:class:`FitterPymoo`.
    
    grid_idx: ``int``
        This is only used so it can be stored in the :py:class:``FitResult``. This method
        itself does not have access to the grid.
    
    dist_transform: ``DistTransform``
        The transform for which to generate parametric fits for. Later, we will save a single
        file per transform, containing all related fits.
    
    the_data: ``NDArray[Shape["*"], Float]``
        The 1-D data used for fitting the RV.
    
    the_data_unique: ``NDArray[Shape["*"], Float]``
        1-D Array of data. In case of continuous data, it is the same as ``the_data``. In case
        of discrete data, the data in this array contains a slight jitter as to make all data
        points unique. Using this data is relevant for conducting statistical goodness of fit
        tests.
    
    :return:
        The :py:class:``FitResult``. If the RV could not be fitted, then the parameters in
        the fitting result will have a value of ``None``. This is so this method does not
        throw exceptions. In case of a failure, no statistical tests are computed, either.
    """
    start = timer()
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    qtype = row.qtype
    fit_continuous = row.type == 'continuous'
    RV: type[Union[rv_continuous, rv_discrete]] = None
    if fit_continuous:
        RV = Continuous_RVs_dict[row.rv]
    else: # pragma: no cover
        # We will not cover testing fitting of discrete RVs.
        RV = Discrete_RVs_dict[row.rv]

    unique_vals = ds.is_qtype_discrete(qtype=qtype) and fit_continuous
    data = the_data_unique if unique_vals else the_data

    ret_dict: FitResult = {}
    ret_dict.update(row.to_dict())
    ret_dict.update(dict(
        grid_idx = grid_idx,
        # Override with a string value.
        dist_transform = dist_transform.name,
        transform_value = transform_value,
        params = None,
        stat_tests = None))

    try:
        fitter = fitter_type(dist=RV)
        params = fitter.fit(data=data, verbose=False)
        params_tuple = tuple(params.values())

        ret_dict.update(dict(params = params))

        dist = RV()
        def temp_cdf(x: NDArray[Shape["*"], Float]):
            x = np.asarray(x).copy()
            min_ = np.min(data)
            max_ = np.max(data)
            x[x < min_] = min_
            x[x > max_] = max_
            return dist.cdf(*(x, *params_tuple))
        temp_ppf = lambda x: dist.ppf(*(x, *params_tuple))

        data_st = data if not unique_vals else np.rint(data) # Remove jitter for test
        st = StatisticalTest(data1=data_st, cdf=temp_cdf, ppf_or_data2=temp_ppf, max_samples=25_000)
        ret_dict.update(dict(stat_tests = dict(st)))
    except Exception: # pragma: no cover
        # print(e)
        # Do nothing at the moment, and allow returning a dict without params and stat_tests
        pass
    finally: # pragma: no cover
        if write_temporary_results:
            end = timer() - start
            print(f'DONE! it took {format(end, "0>5.0f")} seconds ({row.type}), [{row.rv}]')
            with open(f'./results/temp/{grid_idx}_{format(end, "0>5.0f")}', 'wb') as f:
                dump(ret_dict, f)

    return ret_dict
