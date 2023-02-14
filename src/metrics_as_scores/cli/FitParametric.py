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

from metrics_as_scores.__init__ import DATASETS_DIR
from scipy.stats._continuous_distns import norminvgauss_gen, gausshyper_gen, genhyperbolic_gen, geninvgauss_gen, invgauss_gen, studentized_range_gen
from scipy.stats._discrete_distns import nhypergeom_gen, hypergeom_gen



class FitParametricWorkflow(Workflow):
    __doc__ = '''
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
    '''.strip()

    def __init__(self) -> None:
        super().__init__()
        self.use_ds: LocalDataset = None
        self.ds: Dataset = None
        self.df: pd.DataFrame = None
        self.fits_dir: Path = None
        self.selected_rvs_c: list[type[rv_continuous]] = None
        self.selected_rvs_d: list[type[rv_discrete]] = None
        self.use_fitter: type[Fitter] = FitterPymoo
        self.num_cpus = 1
    

    def _select_continuous_rvs(self) -> Iterable[type[rv_continuous]]:
        self.q.print('''
You can now select the continuous random variables that you want to
attempt to fit to the data. Select all in case you intend to re-
distribute and publicize your dataset. It is often not worth to de-
select distributions unless you have a specific reason to do so,
because fitting a continuous distribution is comparatively cheap and
fast.'''.strip())
        recommend_ignore = [norminvgauss_gen, gausshyper_gen, genhyperbolic_gen, geninvgauss_gen, invgauss_gen, studentized_range_gen]
        return self.q.checkbox(message='Select continuous random variables:', choices=[Choice(title=type(rv).__name__, value=type(rv), checked=not type(rv) in recommend_ignore) for rv in Continuous_RVs]).ask()


    def _select_discrete_rvs(self) -> Iterable[type[rv_discrete]]:
        self.q.print('''
You can now select the discrete random variables that you want to
additionally attempt to fit to discrete quantity types. Select all
in case you intend to redistribute and publicize your dataset.
While there are much fewer discrete random variables, their fitting
is dramatically more computationally expensive to compute, since a
global search has to be performed.'''.strip())
        # Note how we will use the RV's type, not the problem's!
        from scipy.stats import _discrete_distns
        recommended_ignore = list(rv.__name__ for rv in [nhypergeom_gen, hypergeom_gen])
        return self.q.checkbox(message='Select discrete random variables:', choices=[Choice(title=dp[0], value=getattr(_discrete_distns, dp[0]), checked=not dp[0] in recommended_ignore) for dp in Discrete_Problems.items()]).ask()
    

    def _get_data_tuples(self, dist_transform: DistTransform, continuous: bool) -> tuple[dict[str, float], dict[str, NDArray[Shape["*"], Float]]]:
        """
        Prepares all required datasets for one distribution transform in parallel,
        either for continuous or discrete data.

        dist_transform: ``DistTransform``
            The chosen distribution transform.
        
        continuous: ``bool``
            Passed forward to :meth:`get_data_tuple()`:
            whether the transform is real-valued or must be converted to integer.

        :rtype: ``tuple[dict[str, float], dict[str, NDArray[Shape["*"], Float]]]``
            Returns two dictionaries.
        
        :return: Two dictionaries, where the keys are the same in either. The values
            in the first are computed ideal values for the selected transform. The
            values in the later are 1-D arrays of the data (the distances).
        """
        res = Parallel(n_jobs=min(self.num_cpus, len(self.ds.quantity_types)))(delayed(get_data_tuple)(ds=self.ds, qtype=qtype, dist_transform=dist_transform, continuous_transform=continuous) for qtype in tqdm(self.ds.quantity_types))
        data_dict = dict([(item[0], item[1]) for sublist in res for item in sublist])
        transform_values_dict = dict([(item[0], item[2]) for sublist in res for item in sublist])
        return (transform_values_dict, data_dict)


    def _fit_parametric(self, dist_transform: DistTransform) -> list[dict[str, Any]]:

        # There are two steps to this:
        # 1) Get all data required
        # 2) Fit
        # Continuous:
        s = 'Performing distribution transforms for: '
        self.print_info(text_normal=s, text_vital='continuous')
        transform_values_dict, data_dict = self._get_data_tuples(dist_transform=dist_transform, continuous=True)
        # Discrete:
        self.print_info(text_normal=s, text_vital='discrete')
        transform_values_discrete_dict, data_discrete_dict = self._get_data_tuples(dist_transform=dist_transform, continuous=False)

        self.print_info(text_normal='', text_vital='Starting fitting of distributions, in randomized order.')
        return generate_parametric_fits(
            ds=self.ds,
            num_jobs=self.num_cpus,
            fitter_type=self.use_fitter,
            dist_transform=dist_transform,
            selected_rvs_c=self.selected_rvs_c,
            selected_rvs_d=self.selected_rvs_d,
            data_dict=data_dict,
            data_discrete_dict=data_discrete_dict,
            transform_values_dict=transform_values_dict,
            transform_values_discrete_dict=transform_values_discrete_dict)



    def fit_parametric(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()
        
        datasets = list(get_local_datasets())
        self.use_ds = self.askt(
            prompt='Select the local dataset you want to generate fits for:',
            options=list([(f'{ds["name"]} [{ds["id"]}] by {", ".join(ds["author"])}', ds) for ds in datasets]))
        self.fits_dir = DATASETS_DIR.joinpath(f'./{self.use_ds["id"]}/fits')

        datafile = DATASETS_DIR.joinpath(f'./{self.use_ds["id"]}/org-data.csv')
        self.print_info(text_normal='Reading original data file: ', text_vital=str(datafile))
        self.df = pd.read_csv(filepath_or_buffer=str(datafile), index_col=False)
        self.df[self.use_ds['colname_type']] = self.df[self.use_ds['colname_type']].astype(str)
        self.df[self.use_ds['colname_context']] = self.df[self.use_ds['colname_context']].astype(str)
        self.ds = Dataset(ds=self.use_ds, df=self.df)

        
        self.selected_rvs_c = list(self._select_continuous_rvs())
        self.print_info(text_normal='Having selected ', text_vital=str(len(self.selected_rvs_c)), end='')
        self.q.print(' continuous random variables.')

        self.selected_rvs_d = list(self._select_discrete_rvs())
        self.print_info(text_normal='Having selected ', text_vital=str(len(self.selected_rvs_d)), end='')
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


        self.q.print(10*'-')
        self.q.print('''
We are ready now to fit random variables. Between each of the transform types
(expectation/mean, median, mode, supremum, infimum) you get the chance to pause
and resume later, and the process will skip existing computed transforms. It is
recommended to run the following on a resourceful computer and to exploit the
highest possible degree of parallelism available. Next, you will be asked how
many cores you would like to use.
        '''.strip())
        max_cpus = cpu_count()
        self.num_cpus = int(self.q.text(f'How many cores should I use (1-{max_cpus})?', default=str(max_cpus), validate=lambda s: isint(s) and int(s) > 0 and int(s) <= max_cpus).ask())

        for dist_transform in list(DistTransform):
            result_file = self.fits_dir.joinpath(f'./pregen_distns_{dist_transform.name}.pickle')
            self.q.print(10*'-')

            if result_file.exists():
                self.print_info(text_normal='Fits already computed for transform: ', text_vital=f'{dist_transform.value} [{dist_transform.name}]')
                continue
            
            self.print_info(text_normal='Generating parametric fits for: ', text_vital=f'{dist_transform.value} [{dist_transform.name}]')
            res = self._fit_parametric(dist_transform=dist_transform)
            with open(file=str(result_file), mode='wb') as fp:
                dump(obj=res, file=fp)
