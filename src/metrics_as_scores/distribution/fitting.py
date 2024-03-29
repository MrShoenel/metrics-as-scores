"""
This module is concerned with fitting distributions to data. It supports
both, discrete and continuous distributions. Fitting is done by a common
helper that unifies the way either type of distribution is fitted.
"""


import numpy as np
from numpy.random import default_rng
from math import pow
from collections.abc import MutableMapping
from typing import Any, Callable, Literal, Sequence, TypedDict, Union
from nptyping import Float, NDArray, Shape
from scipy.stats import _continuous_distns, _discrete_distns, _fit
from inspect import getmembers, isclass
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete

Continuous_RVs: list[rv_continuous] = list(map(lambda tpl: tpl[1], filter(lambda tpl: isinstance(tpl[1], rv_continuous), getmembers(_continuous_distns))))
"""
List of continuous random variables that are supported by `scipy.stats`.
Note this list is a list of instances, rather than types.
"""
Discrete_RVs: list[rv_discrete] = list(map(lambda tpl: tpl[1], filter(lambda tpl: isinstance(tpl[1], rv_discrete), getmembers(_discrete_distns))))
"""
List of discrete random variables that are supported by `scipy.stats`.
Note this list is a list of instances, rather than types.
"""

from metrics_as_scores.distribution import fitting_problems
from metrics_as_scores.distribution.fitting_problems import MixedVariableDistributionFittingProblem
temp = list(filter(lambda rv: hasattr(fitting_problems, f'Fit_{type(rv).__name__}'), Discrete_RVs))
Discrete_Problems: dict[str, type[MixedVariableDistributionFittingProblem]] = { x: y for (x, y) in zip(
    map(lambda rv: type(rv).__name__, temp),
    map(lambda rv: getattr(fitting_problems, f'Fit_{type(rv).__name__}'), temp))}
"""
List of fitting problems used by `pymoo` for fitting discrete distributions.
Metrics As Scores only supports fitting of discrete random variables through
`pymoo` for random variables that have a corresponding problem defined.
However, many, if not most, are covered. In case a problem is missing, the
ordinary fitter can be used (which uses differential evolution).
"""
del temp

from pymoo.termination.default import DefaultTermination
from pymoo.indicators.igd import IGD
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.default import DefaultTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.default import SingleObjectiveSpaceTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.max_time import TimeBasedTermination
from scipy.optimize import differential_evolution
from pymoo.core.result import Result
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from scipy.stats import cramervonmises, cramervonmises_2samp, ks_1samp, ks_2samp, epps_singleton_2samp



# We override this class because of a bug in pymoo.
class DesignSpaceTerminationFixed(DesignSpaceTermination):
    def _delta(self, prev, current):
        if prev.dtype == float and current.dtype == float: # pragma: no cover
            return IGD(current).do(prev)
        else:
            # The error was here: it uses axis=1, but it appears this does not work
            # when using with explicitly defined/types variables as we do.
            return np.mean([np.sum(e != prev).max() / len(e) for e in current])


class SingleObjectiveTermination(DefaultTermination):
    """
    This class is used as termination criterion for the :py:class:`FitterPymoo`.
    """
    def __init__(self, xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=75, max_time: int=600, **kwargs) -> None:
        x = RobustTermination(DesignSpaceTerminationFixed(xtol), period=period)
        cv = RobustTermination(ConstraintViolationTermination(cvtol, terminate_when_feasible=False), period=period)
        f = RobustTermination(SingleObjectiveSpaceTermination(ftol, only_feas=True), period=period)
        super().__init__(x, cv, f, **kwargs)
        if max_time is not None:
            self.mt = TimeBasedTermination(max_time=max_time)
            self.criteria.append(self.mt)



class Fitter:
    """
    This class is used as a generic approach to fit any random variable to any kind
    of data. However, it is recommended to use the more stable :py:class:`FitterPymoo`.
    """

    # All have 'loc' which is an integer, and we will derive it from the data (i.e., use int([min/max](data)))
    Practical_Ranges = dict(
        bernoulli_gen = dict(
            p = [0., 1.]
        ),
        betabinom_gen = dict(
            n = [0, 5_000],
            a = [5e-308, 1e3],
            b = [5e-308, 1e3]
        ),
        binom_gen = dict(
            n = [1, 25_000], # No. of trials
            p = [0., 1.]
        ),
        boltzmann_gen = {
            'lambda': [0., 1e5],
            'N': [1, 25_000]
        },
        dlaplace_gen = dict(
            a = [5e-308, 1e4] # Guessed
        ),
        geom_gen = dict(
            p = [0., 1.]
        ),
        hypergeom_gen = dict(
            # M is the total number of objects, n is total number of Type I objects. The random variate represents the number of Type I objects in N drawn without replacement from the total population.
            M = [1, 25_000],
            n = [0, 25_000],
            N = [0, 25_000]
        ),
        logser_gen = dict(
            p = [0., 1.]
        ),
        nbinom_gen = dict(
            n = [0, 25_000],
            p = [0., 1.]
        ),
        nchypergeom_fisher_gen = dict(
            # Fisher’s noncentral hypergeometric distribution models drawing objects of two types from a bin. M is the total number of objects, n is the number of Type I objects, and odds is the odds ratio: the odds of selecting a Type I object rather than a Type II object when there is only one object of each type. The random variate represents the number of Type I objects drawn if we take a handful of objects from the bin at once and find out afterwards that we took N objects.
            M = [0, 25_000],
            n = [0, 25_000],
            N = [0, 25_000],
            odds = [5e-308, 1e4]
        ),
        nchypergeom_wallenius_gen = dict(
            # Wallenius’ noncentral hypergeometric distribution models drawing objects of two types from a bin. M is the total number of objects, n is the number of Type I objects, and odds is the odds ratio: the odds of selecting a Type I object rather than a Type II object when there is only one object of each type. The random variate represents the number of Type I objects drawn if we draw a pre-determined N objects from a bin one by one.
            M = [0, 25_000],
            n = [0, 25_000],
            N = [0, 25_000],
            odds = [5e-308, 1e4]
        ),
        nhypergeom_gen = dict(
            # Consider a box containing M  balls:, n red and M - n blue. We randomly sample balls from the box, one at a time and without replacement, until we have picked r blue balls. nhypergeom is the distribution of the number of red balls k we have picked.
            M = [0, 25_000],
            n = [0, 25_000],
            r = [0, 25_000],
        ),
        planck_gen = {
            # planck takes  as shape parameter. The Planck distribution can be written as a geometric distribution (geom) with p = 1 - exp(-lambda) shifted by loc = -1.
            # exp(-lambda) get small very quickly, so we choose 100 (~3.7e-44)
            'lambda': [5e-308, 100.]
        },
        poisson_gen = {
            'mu': [0., 1e6]
        },
        randint_gen = dict(
            low = [-25_000, 25_000],
            high = [-25_000, 25_000]
        ),
        skellam_gen = dict(
            mu1 = [5e-308, 5e3],
            mu2 = [5e-308, 5e3]
        ),
        yulesimon_gen = dict(
            alpha = [5e-308, 2e4] # Guessed
        ),
        zipf_gen = dict(
            a = [1. + 1e-12, 2e4] # Guessed
        ),
        zipfian_gen = dict(
            a = [0., 2e4],
            n = [0, 25_000]
        )
    )
    """
    A dictionary of practical bounds for the parameters of discrete distributions.
    It is used by the ``Fitter`` when using differential evolution to optimize the
    fit of a distribution.
    Note that the ``FitterPymoo`` does not use these. Instead, it relies on separate
    problems that are defined for each discrete random variable.
    """

    def __init__(self, dist: type[Union[rv_continuous, rv_discrete]]) -> None:
        """
        dist: ``type[Union[rv_continuous, rv_discrete]]``
            Specify the class of the random variable you want to fit.
        """
        if not issubclass(dist, rv_continuous) and not issubclass(dist, rv_discrete):
            raise Exception('The distribution given is not continuous or discrete random variable.')
        self.dist = dist

        if self.is_discrete and not dist.__name__ in Fitter.Practical_Ranges.keys():
            raise Exception(f'Cannot fit discrete distribution "{dist.__name__}" because no practical ranges exist.')
    
    @staticmethod
    def _optimizer_de(fun, bounds, integrality):
        return differential_evolution(func=fun, bounds=bounds, integrality=integrality, maxiter=10_00)

    @property
    def is_discrete(self) -> bool:
        """Shortcut getter to return whether the used random variable is discrete."""
        return issubclass(self.dist, rv_discrete)

    @property
    def is_continuous(self) -> bool:
        """Shortcut getter to return whether the used random variable is continuous."""
        return issubclass(self.dist, rv_continuous)

    def fit(self, data: NDArray[Shape["*"], Float], **kwargs) -> dict[str, Union[float, int]]:
        """
        Convenience method to fit the random variable. If it is continuous, calls
        :py:meth:`rv_continuous.fit()`, which uses maximum likelihood estimation.
        If it is discrete, uses differential evolution to find an estimate.

        data: ``NDArray[Shape["*"], Float]``
            The data to fit to.
        
        :return:
            A dictionary with named parameters and their values.
        """
        params: tuple[float] = None
        if self.is_continuous:
            dist: type[rv_continuous] = self.dist
            params_tuple = dist().fit(data=data)
            params = { x: y for (x, y) in zip(map(lambda p: p.name, dist()._param_info()), params_tuple) }
        else:
            dist: type[rv_discrete] = self.dist
            bounds = list(Fitter.Practical_Ranges[dist.__name__].values())
            bounds.append([int(np.floor(np.min(data))), int(np.ceil(np.max(data)))]) # for 'loc'
            result: _fit.FitResult = _fit.fit(dist=dist(), data=data, bounds=bounds, optimizer=Fitter._optimizer_de)
            params = result.params._asdict()

        return params



class FitterPymoo(Fitter):
    """
    This class inherits from :py:class:`Fitter` and is the modern successor to it.
    This is because it uses :py:mod:`pymoo` to fit discrete random variables, together
    with a set of specially defined fitting problems.
    """
    def __init__(self, dist: type[Union[rv_continuous, rv_discrete]]) -> None:
        if not isclass(dist):
            raise Exception(f'The given object {dist} is not a subclass of rv_generic.')
        if issubclass(dist, rv_discrete):
            if not dist.__name__ in Discrete_Problems.keys():
                raise Exception('No Pymoo problem available to optimize the random variable.')
            self.problem = Discrete_Problems[dist.__name__]
        else:
            from warnings import warn
            warn("Fitting continuous random variable using variable's fit()-method")

        super().__init__(dist=dist)

    def fit(self, data: NDArray[Shape["*"], Float], max_samples = 10_000, minimize_seeds = [1_337, 0xdeadbeef, 45640321], verbose: bool=True, stop_after_first_res: bool=True) -> dict[str, Union[float, int]]:
        """
        Fits the random variable to the given data. For continuous data, calls :py:meth:`Fitter.fit()`.
        For discrete random variables, however, it solves a mixed-variable problem using a genetic
        algorithm. The documentation below apply to fitting a discrete random variable.

        data: ``NDArray[Shape["*"], Float]``
            The 1-D data to fit the random variable to.
        
        max_samples: ``int``
            Used to deterministically sub-sample the data, should it be longer than this.
        
        minimize_seeds: ``list[int]``
            A list of integer seeds. For each, the optimization is run once, and the result
            with smallest solution is returned.
        
        verbose: ``bool``
            Passed to :py:meth:`minimize()`.
        
        stop_after_first_res: ``bool``
            Whether to stop after the first successful minimization. If `True`, then no further
            fits for the remaining seeds are computed. This is the default as optimization usually
            succeeds and subsequent successful runs are rarely or only insignificantly better.
        
        :raises: Exception:
            If the optimization does not find a single solution.
        
        :rtype: ``dict[str, Union[float, int]]``

        :return:
            A dictionary with parameter names and values of the best found solution.
        """
        if data.shape[0] > max_samples:
            # Then we will sub-sample to speed up the process.
            rng = default_rng(seed=1)
            data = rng.choice(data, size=max_samples, replace=False)

        if self.is_continuous:
            return super().fit(data=data)

        algorithm = MixedVariableGA(pop=50, n_offsprings=2)
        problem = self.problem(data=data)
        results: list[Result] = []
        for seed in minimize_seeds:
            res = minimize(problem, algorithm,
                termination=SingleObjectiveTermination(
                    xtol=1e-8,
                    cvtol=1e-8,
                    ftol=1e-6,
                    period=500,
                    max_time=600,
                    n_max_gen=10_000,
                    n_max_evals=20_000),
                seed=seed, verbose=verbose, save_history=False)
            if res.feas and res.F[0] < float('inf'):
                results.append(res)
                if stop_after_first_res:
                    break # 1 result is enough

        if len(results) > 0:
            results.sort(key=lambda r: r.F[0])
            return results[0].X

        raise Exception('Optimization did not find a maximum likelihood estimate.') # pragma: no cover



StatTest_Types = Literal[
    'cramervonmises_jittered', 'cramervonmises_ordinary',
    'cramervonmises_2samp_jittered', 'cramervonmises_2samp_ordinary',
    'epps_singleton_2samp_jittered', 'epps_singleton_2samp_ordinary',
    'ks_1samp_jittered', 'ks_1samp_ordinary',
    'ks_2samp_jittered', 'ks_2samp_ordinary']


class TestJson(TypedDict):
    """
    Used for serializing a :py:class:`StatisticalTest` to JSON.
    This class represents the result of a single test.
    """
    pval: float
    stat: float

class StatisticalTestJson(TypedDict):
    """
    Used for serializing a :py:class:`StatisticalTest` to JSON.
    This class represents a set of tests and their results.
    """
    tests: dict[str, TestJson]
    discrete_data1: bool
    discrete_data2: bool


class StatisticalTest:
    """
    This class is used to conduct various statistical goodness-of-fit tests. Since not
    every test is always applicable (e.g., the Kolmogorov--Smirnov test is not applicable
    to continuous data), a variety of tests are conducted and the most suitable test (and
    its results) are then typically selected at runtime.
    """
    def __init__(
        self,
        data1: NDArray[Shape["*"], Float],
        cdf: Callable[[Union[float, int]], float],
        ppf_or_data2: Union[NDArray[Shape["*"], Float], Callable[[Union[float, int]], float]],
        data2_num_samples: int=None,
        method = 'auto',
        stat_tests=[cramervonmises, cramervonmises_2samp, ks_1samp, ks_2samp, epps_singleton_2samp],
        max_samples: int=10_000
    ) -> None:
        """
        Initializes a new StatisticalTest. Works with real- and integer-valued data. For the
        latter, additional tests including a deterministic jitter are performed, as these are
        often more representative.

        data1: ``NDArray[Shape["*"], Float]``
            A 1-D array of the data, the first sample.
        
        cdf: ``Callable[[Union[float, int]], float]``
            A CDF used to determine if `data1` comes from it. Used in the one-sample tests.
        
        ppf_or_data2: ``Union[NDArray[Shape["*"], Float], Callable[[Union[float, int]], float]]``
            Either a second 1-D array of the second sample, or a PPF that can be used to
            generate a second sample by inverse sampling.
        
        data2_num_samples: ``int``
            The number of samples to draw for the second sample if `ppf_or_data2` is a PPF.
        
        method: ``str``
            Passed to the statistical tests that support a `method`-parameter.
        
        stat_tests: ``[cramervonmises, cramervonmises_2samp, ks_1samp, ks_2samp, epps_singleton_2samp]``
            List of statistical tests to conduct.
        
        max_samples: ``int``
            The maximum amount of data points in either sample. Also limits `data2_num_samples`.
        """
        data2: NDArray[Shape["*"], Float] = None
        if isinstance(ppf_or_data2, np.ndarray):
            data2 = ppf_or_data2
        elif callable(ppf_or_data2):
            data2 = ppf_or_data2(np.linspace(
                start=1e-16, stop=1. - (1e-16), num=data2_num_samples if data2_num_samples is not None else min(data1.size, max_samples)))
        else:
            raise Exception('"ppf_or_data2" must be 1d np.array or callable.')
        
        if data1.shape[0] > max_samples:
            rng = default_rng(seed=1)
            data1 = rng.choice(data1, size=max_samples, replace=False)
        if data2.shape[0] > max_samples:
            rng = default_rng(seed=2)
            data2 = rng.choice(data2, size=max_samples, replace=False)

        self.discrete_data1 = np.allclose(a = data1, b = np.rint(data1), rtol=1e-10, atol=1e-12)
        self.discrete_data2 = np.allclose(a = data2, b = np.rint(data2), rtol=1e-10, atol=1e-12)

        data1_jittered, data2_jittered = None, None
        if self.discrete_data1:
            temp = abs(np.min(data2))
            expo = 0 if temp == 0 else np.log10(temp)
            rng = default_rng(seed=667788)
            jitter1 = rng.uniform(low=pow(10., expo - 6), high=pow(10., expo - 5), size=data1.size)
            data1_jittered = data1 + jitter1
        else:
            data1_jittered = data1

        if self.discrete_data2:
            temp = abs(np.min(data2))
            expo = 0 if temp == 0 else np.log10(temp)
            rng = default_rng(seed=112233)
            jitter2 = rng.uniform(low=pow(10., expo - 6), high=pow(10., expo - 5), size=data2_num_samples if data2_num_samples is not None else data1.size)
            data2_jittered = data2 + jitter2
        else:
            # Sampling from a continuous PPF creates a de facto continuous data2
            data2_jittered = data2

        self.tests: dict[str, TestJson] = {}
        test_types = ['ordinary', 'jittered']
        for test in stat_tests:
            for tt in test_types:
                use_data1, use_data2 = (data1, data2) if tt == 'ordinary' else (data1_jittered, data2_jittered)
                pval, stat = None, None

                try:
                    if test == cramervonmises:
                        result = cramervonmises(rvs=use_data1, cdf=cdf)
                        pval = result.pvalue
                        stat = result.statistic
                    elif test == cramervonmises_2samp:
                        result = cramervonmises_2samp(x=use_data1, y=use_data2, method=method)
                        pval = result.pvalue
                        stat = result.statistic
                    elif test == ks_1samp:
                        stat, pval = ks_1samp(x=use_data1, cdf=cdf, method=method)
                    elif test == ks_2samp:
                        stat, pval = ks_2samp(data1=use_data1, data2=use_data2, method=method)
                    elif test == epps_singleton_2samp:
                        result = epps_singleton_2samp(x=use_data1, y=use_data2)
                        pval = result.pvalue
                        stat = result.statistic
                except: # pragma: no cover
                    pass # Ignore a failed test; e.g. Epps requires > 4 samples

                # Always create the entry, even for failed tests.
                self.tests[f'{test.__name__}_{tt}'] = dict(pval = pval, stat = stat)
    
    def __iter__(self) -> Sequence[tuple[str, Any]]:
        """
        Implemented so that this instance can be transformed into a dictionary and
        then subsequently serialized to JSON.
        """
        # We can call dict(obj) to get this class as a dictionary
        yield ('tests', self.tests)
        yield ('discrete_data1', self.discrete_data1)
        yield ('discrete_data2', self.discrete_data2)
    
    @staticmethod
    def from_dict(d: MutableMapping[str, Any], key_prefix: str='stat_tests_tests') -> dict[str, float]:
        """
        Re-creates the dictionary of tests from a previously to-JSON serialized instance.
        """
        res: dict[str, float] = {}
        for k in d.keys():
            if k.startswith(key_prefix):
                res[k.replace(key_prefix, '')] = d[k]
        return res
