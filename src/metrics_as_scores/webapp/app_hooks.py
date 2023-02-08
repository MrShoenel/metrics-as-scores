from pathlib import Path
from sys import argv
from pickle import load
from metrics_as_scores.tools.lazy import SelfResetLazy
from metrics_as_scores.distribution.distribution import Empirical, DistTransform, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete, Dataset
from metrics_as_scores.cli.helpers import get_local_datasets, LocalDataset
from metrics_as_scores.webapp import data


this_dir = Path(__file__).resolve().parent
datasets_dir = this_dir.parent.parent.parent.joinpath('./datasets')


def unpickle(file: str):
    try:
        with open(file=file, mode='rb') as f:
            return load(f)
    except Exception as e:
        raise Exception('The webapp relies on precomputed results. Please generate them using the file pregenerate.py before running this webapp.') from e


def load_data(dataset_id: str, preload: bool=False):
    print(f'Attempting to load dataset with ID: {dataset_id}')
    manifest: LocalDataset = None
    temp = list(get_local_datasets())
    for ds in temp:
        if ds['id'] == dataset_id:
            manifest = ds
            break
    
    if manifest is None:
        raise FileNotFoundError(f'There is no local dataset with ID "{dataset_id}".')
    import pandas as pd
    ds = Dataset(ds=manifest, df=pd.DataFrame())
    data.ds = ds
    print(f'Dataset loaded: {ds.ds["desc"]} by {", ".join(ds.ds["author"])}')
    densities_dir = datasets_dir.joinpath(f'./{ds.ds["id"]}/densities')
    data.dataset_dir = datasets_dir.joinpath(f'./{ds.ds["id"]}')


    clazzes = [Empirical, Empirical_discrete, KDE_approx, Parametric, Parametric_discrete]
    transfs = list(DistTransform)

    data.cdfs = {}
    for clazz in clazzes:
        for transf in transfs:
            data.cdfs[f'{clazz.__name__}_{transf.name}'] = SelfResetLazy(reset_after=3600.0 * (4. if preload else 1.),
                fn_create_val=lambda clazz=clazz, transf=transf: unpickle(densities_dir.joinpath(f'./densities_{clazz.__name__}_{transf.name}.pickle')))
            if preload:
                print(f'Pre-loading data for {clazz.__name__}_{transf.name}')
                data.cdfs[f'{clazz.__name__}_{transf.name}'].value
    pass

def on_server_loaded(server_context):
    dataset_id: str = None
    for arg in argv:
        if arg.startswith('dataset='):
            dataset_id = arg.split('=')[1]
            break
    load_data(dataset_id=dataset_id, preload='preload' in argv)
