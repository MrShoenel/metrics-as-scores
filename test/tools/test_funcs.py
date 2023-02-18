import numpy as np
from metrics_as_scores.tools.funcs import nonlinspace



def test_nonlinspace():
    nlp = nonlinspace(start=1.0, stop=10.0, num=100)

    assert nlp.shape[0] == 100
    assert not np.allclose(nlp, np.linspace(start=1.0, stop=10.0, num=nlp.shape[0]))
