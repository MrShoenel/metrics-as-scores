import numpy as np
import pandas as pd
from tempfile import tempdir
from pathlib import Path
from test_Dataset import get_elisa
from metrics_as_scores.data.pregenerate import generate_densities, generate_empirical, generate_empirical_discrete
from metrics_as_scores.distribution.distribution import Empirical, DistTransform



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


def test_generate_empirical():
    ds = get_elisa()
    ds.ds['qtypes'] = { 'Lot1': 'continuous' }
    ds.ds['contexts'] = ['Run1']

    generate_empirical(clazz=Empirical, dataset=ds, transform=DistTransform.NONE, densities_dir=Path(tempdir))


def test_generate_empirical_discrete():
    ds = get_elisa()
    ds.ds['qtypes'] = { 'Lot1': 'continuous' }
    ds.ds['contexts'] = ['Run1']

    generate_empirical_discrete(dataset=ds, densities_dir=Path(tempdir), transform=DistTransform.NONE)

    # Test discrete data:
    ds = get_elisa()
    ds.ds['qtypes'] = { 'Lot1': 'discrete' }
    ds.ds['contexts'] = ['Run1']

    generate_empirical_discrete(dataset=ds, densities_dir=Path(tempdir), transform=DistTransform.NONE)
