import pandas as pd
from pathlib import Path
from os import cpu_count
from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import isint, get_local_datasets
from metrics_as_scores.distribution.distribution import Parametric, Parametric_discrete, DistTransform, Dataset, LocalDataset, Empirical, Empirical_discrete, KDE_approx
from metrics_as_scores.data.pregenerate import generate_parametric, generate_empirical, generate_empirical_discrete
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed


class GenerateDensitiesWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.use_ds: LocalDataset = None
        self.ds: Dataset = None
        self.fits_dir: Path = None
        self.densities_dir: Path = None
        self.num_cpus = 1
    

    def _generate_parametric(self) -> None:
        grid = dict(
            clazz = [Parametric, Parametric_discrete],
            transform = list(DistTransform))
        expanded_grid = pd.DataFrame(ParameterGrid(param_grid=grid))
        Parallel(n_jobs=min(self.num_cpus, len(expanded_grid.index)))(delayed(generate_parametric)(self.ds, self.densities_dir, self.fits_dir, expanded_grid.iloc[i,]['clazz'], expanded_grid.iloc[i,]['transform']) for i in range(len(expanded_grid.index)))
    
    def _generate_empirical_kde(self) -> None:
        grid = dict(
            clazz = [Empirical, KDE_approx],
            transform = list(DistTransform))
        expanded_grid = pd.DataFrame(ParameterGrid(param_grid=grid))
        Parallel(n_jobs=min(self.num_cpus, len(expanded_grid.index)))(delayed(generate_empirical)(self.ds, self.densities_dir, expanded_grid.iloc[i,]['clazz'], expanded_grid.iloc[i,]['transform']) for i in range(len(expanded_grid.index)))
    
    def _generate_empirical_discrete(self) -> None:
        grid = dict(
            clazz = [Empirical_discrete],
            transform = list(DistTransform))
        expanded_grid = pd.DataFrame(ParameterGrid(param_grid=grid))
        Parallel(n_jobs=min(self.num_cpus, len(expanded_grid.index)))(delayed(generate_empirical_discrete)(self.ds, self.densities_dir, expanded_grid.iloc[i,]['transform']) for i in range(len(expanded_grid.index)))

    

    def pre_generate(self) -> None:
        self.q.print('\n' + 10*'-')
        self.q.print('''
This workflow generates density-related functions that are used by the
Web Application. While those can be large, generating them on-the-fly is
usually not possible in acceptable time. Using pre-generated functions
is a trade-off between space and user experience, where we sacrifice the
former as it is cheaper.

For each quantity and each context, we pre-generate functions for the
probability density (PDF), the cumulative distribution (CDF) and its
complement (CCDF), as well as the quantile (or percent point) function
(PPF). So for one quantity and one context, we pre-generate one density
that unites those four functions.

There are 5 primary classes of densities: Parametric, Parametric_discrete,
Empirical, Empirical_discrete, and KDE_approx. Please refer to the
documentation for details about these. Generating parametric densities
uses the computed fits from another workflow, is cheap, fast, and does
not consume much space. KDE_approx makes excessive use of oversampling
(by design), which can result in large files. The empirical densities'
size corresponds to the size of the dataset you are using (although there
is a high limit beyond which sampling will be applied).
'''.strip())
        self.q.print('')
        self.q.print(10*'-')

        datasets = list(get_local_datasets())
        self.use_ds = self.askt(
            prompt='Select the local dataset you want to pre-generate densities for:',
            options=list([(f'{ds["name"]} [{ds["id"]}] by {", ".join(ds["author"])}', ds) for ds in datasets]))
        dataset_dir = DATASETS_DIR.joinpath(f'./{self.use_ds["id"]}')
        self.ds = Dataset(ds=self.use_ds, df=pd.read_csv(str(dataset_dir.joinpath('./org-data.csv'))))
        self.fits_dir = dataset_dir.joinpath(f'./fits')
        self.densities_dir = dataset_dir.joinpath(f'./densities')

        
        self.q.print('''
We are ready now to pre-generate densities. This will require up to approx.
10 GB of space. It is recommended to run the following on a resourceful
computer and to exploit the highest possible degree of parallelism available.'''.strip())
        max_cpus = cpu_count()
        self.num_cpus = int(self.q.text(f'How many cores should I use (1-{max_cpus})?', default=str(max_cpus), validate=lambda s: isint(s) and int(s) > 0 and int(s) <= max_cpus).ask())



        self._generate_parametric()

        self._generate_empirical_kde()

        self._generate_empirical_discrete()
