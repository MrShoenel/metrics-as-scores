import numpy as np
import pandas as pd
from metrics_as_scores.tools.funcs import nonlinspace, natsort, transform_to_MAS_dataset
from sklearn.datasets import load_iris
from pytest import raises



def test_nonlinspace():
    nlp = nonlinspace(start=1.0, stop=10.0, num=100)

    assert nlp.shape[0] == 100
    assert not np.allclose(nlp, np.linspace(start=1.0, stop=10.0, num=nlp.shape[0]))

    # assert it produces a linear space
    linear_space = nonlinspace(start=1.0, stop=1.0 - 1e-30, num=10)
    assert np.allclose(a = linear_space - linear_space[0], b=np.zeros(shape=linear_space.shape), rtol=1e-15, atol=1e-15)


def test_natsort():
    temp = ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    temp.sort(key=natsort)
    assert temp == ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']


def test_transform_to_MAS_dataset():
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame

    res = transform_to_MAS_dataset(df=df, group_col='target', feature_cols=iris.feature_names)

    for cn in ['Value', 'Feature', 'Group']:
        assert cn in res.columns
    
    assert len(res.index) == len(iris.feature_names) * len(df)
    
    with raises(Exception, match='You must select one or more features.'):
        transform_to_MAS_dataset(df=df, group_col='target', feature_cols=[])
    with raises(Exception, match=f'The feature "BLA" is not a column of the given data frame.'):
        transform_to_MAS_dataset(df=df, group_col='BLA', feature_cols=iris.feature_names)
    with raises(Exception, match=f'The feature "foo" is not a column of the given data frame.'):
        transform_to_MAS_dataset(df=df, group_col='target', feature_cols=['foo'])
