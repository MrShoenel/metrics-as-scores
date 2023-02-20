import numpy as np
from metrics_as_scores.tools.funcs import nonlinspace, natsort



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