import numpy as np
from test_Dataset import get_elisa
from metrics_as_scores.data.pregenerate import generate_densities



def test_generate_densities():
    ds = get_elisa()
    ds.ds['qtypes'] = { 'Lot1': 'continuous' }
    ds.ds['contexts'] = ['Run1']
    res = generate_densities(dataset=ds, num_jobs=1)
    assert isinstance(res, dict)

    assert 'Run1_Lot1' in res
    assert '__ALL___Lot1' in res

    r1 = res['Run1_Lot1']
    temp = r1.cdf(np.linspace(start=-20, stop=20, num=100))
    assert np.all((temp >= 0.0) & (temp <= 1.0))
