import numpy as np
import pandas as pd
from re import escape
from pytest import raises
from json import load
from pathlib import Path
from metrics_as_scores.distribution.distribution import Dataset, LocalDataset, DistTransform
from metrics_as_scores.cli.helpers import isnumeric
from metrics_as_scores.tools.funcs import transform_to_MAS_dataset


this_dir = Path(__file__).parent
qcc_manifest_file = this_dir.joinpath('./qcc-manifest.json')
elisa_manifest_file = this_dir.joinpath('./elisa-manifest.json')
elisa_data_file = this_dir.joinpath('./elisa-org-data.csv')
mdhaber_manifest_file = this_dir.joinpath('./mdhaber-manifest.json')
mdhaber_data_file = this_dir.joinpath('./mdhaber-org-data.csv')


def get_elisa() -> Dataset:
    elisa_manifest: LocalDataset = None
    with open(file=str(elisa_manifest_file), mode='r', encoding='utf-8') as fp:
        elisa_manifest = load(fp=fp)
    return Dataset(ds=elisa_manifest, df=pd.read_csv(str(elisa_data_file), index_col=False))


def test_Dataset():
    qcc_manifest: LocalDataset = None
    with open(file=str(qcc_manifest_file), mode='r', encoding='utf-8') as fp:
        qcc_manifest = load(fp=fp)
    # We test the dataset loading the QCC manifest.
    ds = Dataset(ds=qcc_manifest, df=pd.DataFrame())

    qtypes = set(ds.quantity_types)
    assert len(qtypes) == 23
    qtypes_c = set(ds.quantity_types_continuous)
    qtypes_d = set(ds.quantity_types_discrete)

    assert len(qtypes_c) == 5
    for qt in ['LCOM', 'RMA', 'RMD', 'RMI', 'SIX']:
        assert qt in qtypes_c
    
    for qt in qtypes_c.union(qtypes_d):
        assert qt in qtypes
    

    # There are 11 contexts
    contexts = set(ds.contexts(include_all_contexts=False))
    assert len(contexts) == 11
    assert not '__ALL__' in contexts
    contexts = set(ds.contexts(include_all_contexts=True))
    assert len(contexts) == 12
    assert '__ALL__' in contexts

    
    # Check type of ideal values: must be int/float or None:
    for iv in ds.ideal_values.values():
        assert iv is None or isnumeric(iv)
    
    # Check descriptions: None or str:
    for desc in [ds.qytpe_desc(qtype=qty) for qty in qtypes]:
        assert desc is None or isinstance(desc, str)
    for desc in [ds.context_desc(context=ctx) for ctx in ds.contexts(include_all_contexts=False)]:
        assert desc is None or isinstance(desc, str)
    # Also:
    with raises(KeyError):
        ds.context_desc('__ALL__')
    

def test_Dataset_data():
    """
    Here we'll test subsetting the data.
    """
    elisa_manifest: LocalDataset = None
    with open(file=str(elisa_manifest_file), mode='r', encoding='utf-8') as fp:
        elisa_manifest = load(fp=fp)
    # We test the dataset loading the QCC manifest.
    ds = Dataset(ds=elisa_manifest, df=pd.read_csv(str(elisa_data_file), index_col=False))

    # Data has 3 data points per lot and run
    data = ds.data(qtype='Lot1')
    assert data.shape[0] == 15
    data = ds.data(qtype='Lot1', context=None)
    assert data.shape[0] == 15
    data = ds.data(qtype='Lot1', context='__ALL__')
    assert data.shape[0] == 15

    data = ds.data(qtype='Lot2', context='Run1')
    assert data.shape[0] == 3

    with raises(Exception, match='The context "FOOBAR" is not known.'):
        ds.data(qtype='Lot1', context='FOOBAR')
    
    # test sub-sampling:
    data = ds.data(qtype='Lot5', sub_sample=10)
    assert data.shape[0] == 10


def test_Dataset_transform():
    ds = get_elisa()
    assert ds.has_sufficient_observations()

    data = ds.data(qtype='Lot1')
    data_d = np.rint(10.0 * data)

    _, temp = ds.transform(data=data, dist_transform=DistTransform.NONE)
    assert np.allclose(data, temp)

    # Here we just make sure the transforms all work.
    for dt in list(DistTransform):
        if dt == DistTransform.NONE:
            continue
        for cv in [True, False]:
            use_data = data if cv else data_d
            _, temp = ds.transform(data=use_data, dist_transform=dt, continuous_value=cv)


def test_Dataset_stat_tests():
    ds = get_elisa()
    qtypes = ds.quantity_types
    contexts_all = list(ds.contexts(include_all_contexts=True))
    
    test_anova = ds.analyze_groups(use='anova', qtypes=qtypes, contexts=contexts_all)
    assert isinstance(test_anova, pd.DataFrame)
    # We get one line for each qtype:
    assert len(test_anova.index) == len(qtypes)
    assert ','.join(test_anova.columns) == 'qtype,stat,pval,across_contexts'
    row = test_anova.iloc[0,:]
    assert row.across_contexts == ';'.join(contexts_all)

    with raises(Exception, match='Requires one or quantity types and two or more contexts.'):
        ds.analyze_groups(use='anova', qtypes=[], contexts=contexts_all)
    with raises(Exception, match='Requires one or quantity types and two or more contexts.'):
        ds.analyze_groups(use='anova', qtypes=qtypes, contexts=contexts_all[0:1])
    

    test_kruskal = ds.analyze_groups(use='kruskal', qtypes=qtypes, contexts=contexts_all)
    assert isinstance(test_kruskal, pd.DataFrame)
    assert len(test_kruskal.index) == len(qtypes)
    assert ','.join(test_kruskal.columns) == 'qtype,stat,pval,across_contexts'
    row = test_kruskal.iloc[0,:]
    assert row.across_contexts == ';'.join(contexts_all)

    
    test_tukey = ds.analyze_TukeyHSD(qtypes=qtypes)
    assert isinstance(test_tukey, pd.DataFrame)
    num_per_qty = np.sum(range(len(contexts_all)))
    assert len(test_tukey.index) == num_per_qty * len(qtypes)
    assert ','.join(test_tukey.columns) == 'group1,group2,meandiff,p-adj,lower,upper,reject'

    with raises(Exception, match='Requires one or quantity types.'):
        ds.analyze_TukeyHSD(qtypes=[])
    

    for use_ks2 in [True, False]:
        test_distr = ds.analyze_distr(qtypes=qtypes, use_ks_2samp=use_ks2)
        assert ','.join(test_distr.columns) == 'qtype,stat,pval,group1,group2'
        assert len(test_distr.index) == num_per_qty * len(qtypes)
    
    with raises(Exception, match='Requires one or more quantity types.'):
        ds.analyze_distr(qtypes=[])


def test_mdhaber_dataset():
    mdhaber_manifest: LocalDataset = None
    with open(file=str(mdhaber_manifest_file), mode='r', encoding='utf-8') as fp:
        mdhaber_manifest = load(fp=fp)
    
    df = transform_to_MAS_dataset(df=pd.read_csv(str(mdhaber_data_file), index_col=False), group_col='ID', feature_cols=['Feature 1', 'Feature 2'])
    ds = Dataset(ds=mdhaber_manifest, df=df)

    assert False == ds.has_sufficient_observations(raise_if_not=False)

    with raises(Exception, match=escape(f'The quantity type "Feature 1" in context "1" has insufficient (1) observation(s).')):
        # It should throw using the first feature in the first context already
        ds.has_sufficient_observations(raise_if_not=True)
