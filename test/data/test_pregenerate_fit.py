import numpy as np
import pandas as pd
from test_Dataset import get_elisa
from metrics_as_scores.data.pregenerate_fit import fit, get_data_tuple
from metrics_as_scores.distribution.distribution import DistTransform
from metrics_as_scores.distribution.fitting import FitterPymoo




def get_fit_row():
    return pd.DataFrame({
        'qtype': ['Lot1'],
        'type': ['continuous'],
        'rv': ['norm_gen']
    })



def test_fit():
    ds = get_elisa()

    row = get_fit_row().iloc[0,:]
    res = fit(ds=ds, fitter_type=FitterPymoo, grid_idx=0, row=row, dist_transform=DistTransform.NONE, the_data=ds.data(qtype='Lot1'), the_data_unique=ds.data(qtype='Lot1'), transform_value=None)

    assert isinstance(res, dict)
    assert isinstance(res['params'], dict)



def test_get_data_tuple():
    ds = get_elisa()

    contexts_all = list(ds.contexts(include_all_contexts=True))
    l = get_data_tuple(ds=ds, qtype='Lot1', dist_transform=DistTransform.INFIMUM)
    assert len(l) == len(contexts_all) * 2

    # Test some data:
    l0 = l[0]
    assert l0[0] == f'{contexts_all[0]}_Lot1_u'
    assert abs(l0[2] - np.min(ds.data(qtype='Lot1', context=list(ds.contexts())[0]))) <= 1e-12

