from typing import Any, Iterable
from pathlib import Path
from nptyping import Float, NDArray, Shape
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
from metrics_as_scores.cli.helpers import get_local_datasets, isint
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.distribution.distribution import DistTransform, LocalDataset, Dataset
from metrics_as_scores.distribution.fitting import Fitter, FitterPymoo, Continuous_RVs, Discrete_Problems
from metrics_as_scores.data.pregenerate_fit import get_data_tuple
from metrics_as_scores.data.pregenerate_distns import generate_parametric_fits
from questionary import Choice
from pickle import dump
from os import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

from scipy.stats._continuous_distns import norminvgauss_gen, gausshyper_gen, genhyperbolic_gen, geninvgauss_gen, invgauss_gen, studentized_range_gen
from scipy.stats._discrete_distns import nhypergeom_gen, hypergeom_gen


this_dir = Path(__file__).resolve().parent
datasets_dir = this_dir.parent.parent.parent.joinpath('./datasets')



class FitParametricWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.use_ds: LocalDataset = None
        self.df: pd.DataFrame = None
        self.fits_dir: Path = None
        self.use_fitter: type[Fitter] = FitterPymoo
    

    def _select_continuous_rvs(self) -> Iterable[rv_continuous]:
        self.q.print('''
You can now select the continuous random variables that you want to
attempt to fit to the data. Select all in case you intend to re-
distribute and publicize your dataset. It is often not worth to de-
select distributions unless you have a specific reason to do so,
because fitting a continuous distribution is comparatively cheap and
fast.'''.strip())
        return self.q.checkbox(message='Select continuous random variables:', choices=[Choice(title=type(rv).__name__, value=rv, checked=True) for rv in Continuous_RVs]).ask()
    
    def _select_discrete_rvs(self) -> Iterable[rv_discrete]:
        self.q.print('''
You can now select the discrete random variables that you want to
additionally attempt to fit to discrete quantity types. Select all
in case you intend to redistribute and publicize your dataset.
While there are much fewer discrete random variables, their fitting
is dramatically more computationally expensive to compute, since a
global search has to be performed.'''.strip())
        return self.q.checkbox(message='Select discrete random variables:', choices=[Choice(title=dp[0], value=dp[1], checked=True) for dp in Discrete_Problems.items()]).ask()
    

    def _iterable_fits(self, rvs: Iterable[Union[rv_continuous, rv_discrete]]) -> Iterable[tuple[Path, str, str, Union[rv_continuous, rv_discrete]]]:
        from itertools import product
        for comb in product(self.use_ds['qtypes'].keys(), self.use_ds['contexts'], rvs):
            fit_file = f'{comb[0]}_{comb[1]}_{"c" if issubclass(type(comb[2]), rv_continuous) else "d"}_{type(comb[2]).__name__}.json'
            fit_path = self.fits_dir.joinpath(f'./{fit_file}')
            if fit_path.exists():
                self.print_info(text_normal='Fit already exists, skipping: ', text_vital=fit_file)
            else:
                yield (fit_path, comb[0], comb[1], comb[2])
    

    def _fit_continuous_rvs(self, rvs: Iterable[rv_continuous]) -> None:
        """
        For each quantity type, and for each context, fit each random variable
        once, and save the result in "`qtype_context_rv`". If that file exists,
        skip to next.
        """
        perms = list(self._iterable_fits(rvs=rvs))
        with Progress() as progress:
            task1 = progress.add_task('[cyan]Fitting continuous distributions...', total=len(perms))
            for fit_path, qtype, ctx, rv in perms:
                data = self.df[(self.df[self.use_ds['colname_type']] == qtype) & (self.df[self.use_ds['colname_context']] == ctx)][self.use_ds['colname_data']].to_numpy()
                fitter = self.use_fitter(dist=type(rv))
                res = fitter.fit(data=data)
                with open(file=str(fit_path.resolve()), mode='w', encoding='utf-8') as fp:
                    dump(obj=res, fp=fp)
                progress.update(task_id=task1, advance=1)
    

    def fit_parametric(self) -> None:
        self.q.print('\n' + 10*'-')
        self.q.print('''
This workflow fits distributions to an existing dataset. For each
type of quantity, and for each context, a large number of random
variables are fit, and a number of statistical tests are carried
out such that the best-fitting distribution may be selected/used.
Regardless of whether a quantity is continuous or discrete, many
continuous random variables are attempted to fit. If a quantity
is discrete, however, an additional set of discrete random variables
is attempted to fit. Especially the latter might be extraordinarily
expensive.
Therefore, you may only select a subset of random variables that
you want to attempt to fit. However, if you intend to share your
dataset and make it available to others, then you should include
and attempt to fit all distributions.
The following process, once begun, will save the result of fitting
a single type of quantity (from within a single context) as a
separate file. If the file already exists, no new fit is attempted.
This is so that this process can be interrupted and resumed.
'''.strip())
        self.q.print('')
        datasets = list(get_local_datasets())
        self.use_ds = self.askt(
            prompt='Select the local dataset you want to generate fits for:',
            options=list([(f'{ds["name"]} [{ds["id"]}] by {", ".join(ds["author"])}', ds) for ds in datasets]))
        self.fits_dir = datasets_dir.joinpath(f'./{self.use_ds["id"]}/fits')

        datafile = datasets_dir.joinpath(f'./{self.use_ds["id"]}/org-data.csv')
        self.print_info(text_normal='Reading original data file: ', text_vital=str(datafile))
        self.df = pd.read_csv(filepath_or_buffer=str(datafile), index_col=False)
        self.df[self.use_ds['colname_type']] = self.df[self.use_ds['colname_type']].astype(str)
        self.df[self.use_ds['colname_context']] = self.df[self.use_ds['colname_context']].astype(str)

        
        selected_rvs_c = self._select_continuous_rvs()
        self.print_info(text_normal='Having selected ', text_vital=str(len(selected_rvs_c)), end='')
        self.q.print(' continuous random variables.')

        selected_rvs_d = self._select_discrete_rvs()
        self.print_info(text_normal='Having selected ', text_vital=str(len(selected_rvs_d)), end='')
        self.q.print(' discrete random variables.')


        self.q.print(10*'-')
        self.q.print('''
You need to choose how fits should be computed. Metrics As Scores offers
two classes: Fitter and FitterPymoo, where the latter is recommended. As
far as continuous random variables are concerned, it does not make a
difference. For discrete random variables, however, the original Fitter
uses scipy's approach of differential evolution, which does not always
scale well. The class FitterPymoo, on the other hand, uses separate, mixed
variable genetic algorithm problems, tailored to each random variable.
        '''.strip())
        self.use_fitter = self.askt(options=[
            (f'{FitterPymoo.__name__} [Recommended]', FitterPymoo),
            (f'{Fitter.__name__}', Fitter)
        ], prompt='Which fitter do you want to use?')


        self._fit_continuous_rvs(rvs=selected_rvs_c)

        pass