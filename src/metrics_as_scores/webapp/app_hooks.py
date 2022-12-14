from os import getcwd
from os.path import join
from sys import path, argv
from pathlib import Path
path.append(f'{Path(join(getcwd(), "src")).resolve()}')


from pickle import load
from metrics_as_scores.tools.lazy import SelfResetLazy
from metrics_as_scores.distribution.distribution import Empirical, DistTransform, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete

from . import data


def unpickle(file: str):
    try:
        with open(file=file, mode='rb') as f:
            return load(f)
    except Exception as e:
        raise Exception('The webapp relies on precomputed results. Please generate them using the file pregenerate.py before running this webapp.') from e


def load_data(preload: bool=False):
    print('Loading data')
    clazzes = [Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete]
    transfs = list(DistTransform)

    data.cdfs = {}
    for clazz in clazzes:
        for transf in transfs:
            data.cdfs[f'{clazz.__name__}_{transf.name}'] = SelfResetLazy(reset_after=3600.0 * (4. if preload else 1.),
                fn_create_val=lambda clazz=clazz, transf=transf: unpickle(f'./results/densities_{clazz.__name__}_{transf.name}.pickle'))
            if preload:
                print(f'Pre-loading data for {clazz.__name__}_{transf.name}')
                data.cdfs[f'{clazz.__name__}_{transf.name}'].value
    pass

def on_server_loaded(server_context):
    load_data(preload='preload' in argv)
