import pandas as pd
import numpy as np
import pickle
from abc import ABC
from typing import Callable, Iterable, Literal, Union, TypedDict
from typing_extensions import Self
from nptyping import NDArray, Shape, Float
from metrics_as_scores.distribution.fitting import StatisticalTest
from metrics_as_scores.tools.funcs import cdf_to_ppf
from statsmodels.distributions import ECDF as SMEcdf
from scipy.stats import gaussian_kde, f_oneway, mode
from scipy.integrate import quad
from scipy.optimize import direct
from scipy.stats._distn_infrastructure import rv_generic, rv_continuous
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from strenum import StrEnum
from metrics_as_scores.distribution.fitting import StatTest_Types




class DistTransform(StrEnum):
    NONE = '<none>'
    EXPECTATION = 'E[X] (expectation)'
    MEDIAN = 'Median (50th percentile)'
    MODE = 'Mode (most likely value)'
    INFIMUM = 'Infimum (min. observed value)'
    SUPREMUM = 'Supremum (max. observed value)'



class JsonDataset(TypedDict):
    name: str
    desc: str
    id: str
    author: list[str]
    ideal_values: dict[str, Union[int, float]]


class LocalDataset(JsonDataset):
    origin: str
    colname_data: str
    colname_type: str
    colname_context: str
    qtypes: dict[str, Literal['continuous', 'discrete']]
    contexts: list[str]


class KnownDataset(JsonDataset):
    info_url: str
    download: str



class Density(ABC):
    def __init__(self, range: tuple[float, float], pdf: Callable[[float], float], cdf: Callable[[float], float], ppf: Callable[[float], float]=None, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self.range = range
        self._pdf = pdf
        self._cdf = cdf
        self._ppf = ppf
        self._ideal_value = ideal_value
        self._dist_transform = dist_transform
        self._transform_value: float = None
        self._qtype = qtype
        self._context = context
        self._practical_domain: tuple[float, float] = None
        self._practical_range_pdf: tuple[float, float] = None

        self.transform_value = transform_value

        self.pdf = np.vectorize(self._pdf)
        self.cdf = np.vectorize(self._min_max)
        if ppf is None:
            self.ppf = lambda *args: exec('raise(NotImplementedError())')
        else:
            self.ppf = np.vectorize(self._ppf)
    

    @property
    def qtype(self) -> Union[str, None]:
        return self._qtype
    
    @property
    def context(self) -> Union[str, None]:
        return self._context

    @property
    def ideal_value(self) -> Union[float, None]:
        return self._ideal_value
    
    @property
    def is_quality_score(self) -> bool:
        return self.ideal_value is not None

    @property
    def dist_transform(self) -> DistTransform:
        return self._dist_transform

    @property
    def transform_value(self) -> Union[float, None]:
        return self._transform_value
    
    @transform_value.setter
    def transform_value(self, value: Union[float, None]) -> Self:
        self._transform_value = value
        return self


    def _min_max(self, x: float) -> float:
        if x < self.range[0]:
            return 0.0
        elif x > self.range[1]:
            return 1.0
        return self._cdf(x)
    

    def compute_practical_domain(self, cutoff: float=0.995) -> tuple[float, float]:
        def obj_lb(x):
            return np.square(self.cdf(x) - (1. - cutoff))
        def obj_ub(x):
            return np.square(self.cdf(x) - cutoff)

        r = self.range
        m_lb = direct(func=obj_lb, bounds=(r,), f_min=0.)
        m_ub = direct(func=obj_ub, bounds=(r,), f_min=0.)
        return (m_lb.x[0], m_ub.x[0])

    
    @property
    def practical_domain(self) -> tuple[float, float]:
        if self._practical_domain is None:
            self._practical_domain = self.compute_practical_domain()
        return self._practical_domain
    

    def compute_practical_range_pdf(self) -> tuple[float, float]:
        def obj(x):
            return -1. * np.log(1. + self.pdf(x))

        m = direct(func=obj, bounds=(self.range,), locally_biased=False)#, maxiter=15)
        return (0., self.pdf(m.x[0])[0])
    

    @property
    def practical_range_pdf(self) -> tuple[float, float]:
        if self._practical_range_pdf is None:
            self._practical_range_pdf = self.compute_practical_range_pdf()
        return self._practical_range_pdf

    
    def __call__(self, x: Union[float, list[float], NDArray[Shape["*"], Float]]) -> NDArray[Shape["*"], Float]:
        if np.isscalar(x) or isinstance(x, list):
            x = np.asarray(x)
        return self.cdf(x)
    
    def save_to_file(self, file: str) -> None:
        with open(file=file, mode='wb') as f:
            pickle.dump(obj=self, file=f)
    
    @staticmethod
    def load_from_file(file: str) -> 'Density':
        with open(file=file, mode='rb') as f:
            return pickle.load(file=f)


class KDE_integrate(Density):
    def __init__(self, data: NDArray[Shape["*"], Float], ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self._kde = gaussian_kde(dataset=np.asarray(data))
        lb, ub = np.min(data), np.max(data)
        ext = np.max(data) - lb

        def pdf(x):
            return self._kde.evaluate(points=np.asarray(x))
        
        def cdf(x):
            y, _ = quad(func=pdf, a=self.range[0], b=x)
            return y
        
        m_lb = direct(func=pdf, bounds=((lb - ext, lb),), f_min=1e-6)
        m_ub = direct(func=pdf, bounds=((ub, ub + ext),), f_min=1e-6)
        
        ppf = cdf_to_ppf(cdf=np.vectorize(cdf), x=data, y_left=np.min(data), y_right=np.max(data))

        super().__init__(range=(m_lb.x, m_ub.x), pdf=pdf, cdf=cdf, ppf=ppf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, kwargs=kwargs)


class KDE_approx(Density):
    def __init__(self, data: NDArray[Shape["*"], Float], resample_samples: int=200_000, compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        # First, we'll fit an extra KDE for an approximate PDF.
        # It is used to also roughly estimate its mode.
        rng = np.random.default_rng(seed=1)
        data_pdf = data if data.shape[0] <= 10_000 else rng.choice(a=data, size=10_000, replace=False)
        self._kde_for_pdf = gaussian_kde(dataset=data_pdf)

        self._range_data = (np.min(data), np.max(data))   
        self._kde = gaussian_kde(dataset=np.asarray(data))
        data = self._kde.resample(size=resample_samples, seed=1).reshape((-1,))
        self._ecdf = SMEcdf(x=data)
        self._ppf_interp = cdf_to_ppf(cdf=np.vectorize(self._ecdf), x=data, y_left=np.min(data), y_right=np.max(data))

        super().__init__(range=(np.min(data), np.max(data)), pdf=self._pdf_from_kde, cdf=self._ecdf, ppf=self._ppf_from_ecdf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, kwargs=kwargs)

        self.stat_test = StatisticalTest(data1=data, cdf=self._ecdf, ppf_or_data2=self._ppf_from_ecdf)

        if compute_ranges:
            self.practical_domain
            self.practical_range_pdf
    
    def _pdf_from_kde(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self._kde_for_pdf.evaluate(points=np.asarray(x))
    
    def _ppf_from_ecdf(self, q: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self._ppf_interp(q)
    
    @property
    def pval(self) -> float:
        return self.stat_test.tests['ks_2samp_jittered']['pval']
    
    @property
    def stat(self) -> float:
        return self.stat_test.tests['ks_2samp_jittered']['stat']


class Empirical(Density):
    def __init__(self, data: NDArray[Shape["*"], Float], compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self._ecdf = SMEcdf(data)
        self._ppf_interp = cdf_to_ppf(cdf=np.vectorize(self._ecdf), x=data, y_left=np.min(data), y_right=np.max(data))

        super().__init__(range=(np.min(data), np.max(data)), pdf=gaussian_kde(dataset=data).pdf, cdf=self._ecdf, ppf=self._ppf_from_ecdf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, kwargs=kwargs)

        if compute_ranges:
            self.practical_domain
    
    def _ppf_from_ecdf(self, q: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self._ppf_interp(q)


class Empirical_discrete(Empirical):
    def __init__(self, data: NDArray[Shape["*"], Float], ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self._data_valid = not (data.shape[0] == 1 and np.isnan(data[0]))
        data_int = np.rint(data).astype(int)
        if self._data_valid and not np.allclose(a=data, b=data_int, rtol=1e-10, atol=1e-12):
            raise Exception('The data does not appear to be integral.')

        self._unique, self._counts = np.unique(data_int, return_counts=True)
        self._unique, self._counts = self._unique.astype(int), self._counts.astype(int)
        self._idx: dict[int, int] = { self._unique[i]: self._counts[i] for i in range(self._unique.shape[0]) }

        if self._data_valid:
            super().__init__(data=data_int, compute_ranges=False, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, **kwargs)
        else:
            self._dist_transform = dist_transform
            self._transform_value = transform_value
            self._qtype = qtype
            self._context = context
            self._cdf = self.cdf = self._non_fit_cdf_ppf
            self._ppf = self.ppf = self._non_fit_cdf_ppf

        self._pdf = self._pmf = self._pmf_from_frequencies
        self.pdf = self.pmf = np.vectorize(self._pdf)

        self._practical_domain = (np.min(data_int), np.max(data_int))
        self._practical_range_pdf = (0., float(np.max(self._counts)) / float(np.sum(self._counts)))

    @property
    def is_fit(self) -> bool:
        return self._data_valid
    
    def _non_fit_cdf_ppf(self, x) -> NDArray[Shape["*"], Float]:
        x = np.asarray([x] if np.isscalar(x) else x)
        return np.asarray([np.nan] * x.shape[0])


    def _pmf_from_frequencies(self, x: int) -> float:
        x = int(x)
        if not x in self._idx.keys():
            return 0.

        return float(self._idx[x]) / float(np.sum(self._counts))

    @staticmethod
    def unfitted(dist_transform: DistTransform) -> 'Empirical_discrete':
        return Empirical_discrete(data=np.asarray([np.nan]), dist_transform=dist_transform)


class Parametric(Density):
    def __init__(self, dist: rv_generic, dist_params: tuple, range: tuple[float, float], stat_tests: dict[str, float], use_stat_test: StatTest_Types='ks_2samp_jittered', compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self.dist: Union[rv_generic, rv_continuous] = dist
        self.stat_tests = stat_tests
        self._use_stat_test = use_stat_test
        self.dist_params = dist_params

        super().__init__(range=range, pdf=self.pdf, cdf=self.cdf, ppf=self.ppf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, **kwargs)

        if compute_ranges:
            self.practical_domain
            self.practical_range_pdf
    
    @staticmethod
    def unfitted(dist_transform: DistTransform) -> 'Parametric':
        from scipy.stats._continuous_distns import norm_gen
        return Parametric(dist=norm_gen(), dist_params=None, range=(np.nan, np.nan), stat_tests={}, dist_transform=dist_transform)

    @property
    def use_stat_test(self) -> StatTest_Types:
        return self._use_stat_test
    
    @use_stat_test.setter
    def use_stat_test(self, st: StatTest_Types) -> Self:
        self._use_stat_test = st
        return self
    
    @property
    def pval(self) -> float:
        if not self.is_fit:
            raise Exception('Cannot return p-value for non-fitted random variable.')
        return self.stat_tests[f'{self.use_stat_test}_pval']
    
    @property
    def stat(self) -> float:
        if not self.is_fit:
            raise Exception('Cannot return statistical test statistic for non-fitted random variable.')
        return self.stat_tests[f'{self.use_stat_test}_stat']
    
    @property
    def is_fit(self) -> bool:
        return not self.dist_params is None and not np.any(np.isnan(self.dist_params))
    
    @property
    def practical_domain(self) -> tuple[float, float]:
        if not self.is_fit:
            return (0., 0.)
        return super().practical_domain
    
    @property
    def practical_range_pdf(self) -> tuple[float, float]:
        if not self.is_fit:
            return (0., 0.)
        return super().practical_range_pdf
    
    @property
    def dist_name(self) -> str:
        return self.dist.__class__.__name__
    
    def pdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.pdf(*(x, *self.dist_params)).reshape((x.size,))
    
    def cdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.cdf(*(x, *self.dist_params)).reshape((x.size,))
    
    def ppf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.ppf(*(x, *self.dist_params)).reshape((x.size,))


class Parametric_discrete(Parametric):
    def pmf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.pmf(*(x, *self.dist_params)).reshape((x.size,))
        
    def pdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self.pmf(x=x)
    
    @staticmethod
    def unfitted(dist_transform: DistTransform) -> 'Parametric_discrete':
        from scipy.stats._continuous_distns import norm_gen
        return Parametric_discrete(dist=norm_gen(), dist_params=None, range=(np.nan, np.nan), stat_tests={}, dist_transform=dist_transform)



class Dataset:
    """
    This class encapsulates a local (self created) dataset and provides help with transforming
    it, as well as giving some convenience getters.
    """
    def __init__(self, ds: LocalDataset, df: pd.DataFrame) -> None:
        self.ds = ds
        self.df = df
    
    @property
    def quantity_types(self) -> list[str]:
        return list(self.ds['qtypes'].keys())


    def contexts(self, include_all_domain: bool=False) -> Iterable[str]:
        yield from self.ds['contexts']
        if include_all_domain:
            yield '__ALL__'
    
    def is_qtype_discrete(self, qtype: str) -> bool:
        return self.ds['qtypes'][qtype] == 'discrete'
    
    @property
    def quantity_types_continuous(self) -> list[str]:
        return list(filter(lambda qtype: not self.is_qtype_discrete(qtype=qtype), self.quantity_types))

    @property
    def quantity_types_discrete(self) -> list[str]:
        return list(filter(lambda qtype: self.is_qtype_discrete(qtype=qtype), self.quantity_types))

    def data(self, qtype: str, context: str=None, unique_vals: bool=True, sub_sample: int=None) -> NDArray[Shape["*"], Float]:
        """
        This method is used to select a subset of the data, that is specific to at least
        a type of quantity, and optionally to a context, too.
        """
        new_df = self.df[self.df[self.ds['colname_type']] == qtype]
        if context is not None:
            new_df = new_df[new_df[self.ds['colname_context']] == context]
        
        vals = new_df[self.ds['colname_data']]
        if unique_vals:
            rng = np.random.default_rng(seed=1_337)
            r = rng.choice(a=np.linspace(1e-8, 1e-6, vals.size), size=vals.size, replace=False)
            # Add small but insignificant perturbations to the data to produce unique
            # values that would otherwise be eliminated by certain methods.
            vals += r
        
        vals = vals.to_numpy()
        
        if sub_sample is not None and sub_sample < vals.size:
            rng = np.random.default_rng(seed=1_338)
            vals = rng.choice(a=vals, size=sub_sample, replace=False)
        
        return vals
    
    @staticmethod
    def transform(data: NDArray[Shape["*"], Float], dist_transform: DistTransform=DistTransform.NONE, continuous_value: bool=True) -> tuple[float, NDArray[Shape["*"], Float]]:
        r"""
        Transforms a distribution using an ideal value. The resulting data, therefore,
        is a distribution of distances from the designated ideal value.

        Given a distribution :math:`X` and an ideal value :math:`i`, the distribution of
        distances is defined as :math:`D=\left|X-i\right|`.

        data: ``NDArray[Shape["*"], Float]``
            1-D array of float data, the data to be transformed. The data may also hold
            integers (or floats that are practically integers).
        
        dist_transform: ``DistTransform``
            The transform to apply. If ``DistTransform.NONE``, the data is returned as is,
            ``None`` as the transform value. Any of the other transforms are determined
            from the data (see notes).
        
        continuous_value: ``bool``
            Whether or not the to be determined ideal value should be continuous or not.
            For example, if using the expectation (mean) as transform, even for a discrete
            distribution, this is likely to be a float. Setting ``continuous_value`` to
            ``False`` will round the found mean to the nearest integer, such that the
            resulting distribution :math:`D` is of integral nature, too.

        :rtype: ``tuple[float, NDArray[Shape["*"], Float]]``

        :return: A tuple holding the applied transform value (if the chosen transform was
            not ``DistTransform.NONE``) and the array of distances.
        
        Notes
        -----
        The expectation (mean), in the continuous case, is determined by estimating a
        Gaussian kernel using ``gaussian_kde``, and then integrating it using
        :meth:`Density.practical_domain`. In the discrete case, we use the rounded mean
        of the data.
        Mode and median are similarly computed in the continuous and discrete cases,
        except for the discrete mode we use :meth:`scipy.stats.mode`.
        Supremum and infimum are simply computed (and rounded in the discrete case) from
        the data.
        """
        if dist_transform == DistTransform.NONE:
            return (None, data)

        # Do optional transformation
        transform_value: float=None
        if dist_transform == DistTransform.EXPECTATION:
            if continuous_value:
                temp = KDE_approx(data=data, compute_ranges=True)
                ext = temp.practical_domain[1] - temp.practical_domain[0]
                transform_value, _ = quad(func=lambda x: x * temp._pdf_from_kde(x), a=temp.practical_domain[0] - ext, b=temp.practical_domain[1] + ext, limit=250)
            else:
                # For non-continuous transforms, we should first round the data,
                # because it may contain a jitter (i.e., we expect the data to
                # be integral, but it may has a jitter applied).
                transform_value = np.mean(np.rint(data))
        elif dist_transform == DistTransform.MODE:
            if continuous_value:
                temp = KDE_approx(data=data, compute_ranges=True)
                m = direct(func=lambda x: -1. * np.log(1. + temp._pdf_from_kde(x)), bounds=(temp.range,), locally_biased=False)
                transform_value = m.x[0] # x of where the mode is (i.e., not f(x))!
            else:
                transform_value = mode(a=np.rint(data), keepdims=False).mode
        elif dist_transform == DistTransform.MEDIAN:
            if continuous_value:
                # We'll get the median from the smoothed PDF in order to also get a more smooth value
                temp = KDE_approx(data=data, compute_ranges=True)
                transform_value = np.median(temp._kde.resample(size=50_000, seed=2))
            else:
                transform_value = np.median(a=np.rint(data))
        elif dist_transform == DistTransform.INFIMUM:
            if continuous_value:
                transform_value = np.min(data)
            else:
                transform_value = np.min(np.rint(data))
        elif dist_transform == DistTransform.SUPREMUM:
            if continuous_value:
                transform_value = np.max(data)
            else:
                transform_value = np.max(np.rint(data))
        
        # Now do the convex transform: Compute the distance to the transform value!
        if transform_value is not None:
            if not continuous_value:
                transform_value = np.rint(transform_value)
            data = np.abs(data - transform_value)

        return (transform_value, data)


    def analyze_ANOVA(self, qtypes: Iterable[str], contexts: Iterable[str], unique_vals: bool=True) -> pd.DataFrame:
        """
        For each given type of quantity, this method performs an ANOVA across all
        given contexts.

        qtypes: ``Iterable[str]``
            An iterable of quantity types to conduct the analysis for. For each given
            type, a separate analysis is performed and the result appended to the
            returned data frame.
        
        contexts: ``Iterable[str]``
            An iterable of contexts across which each of the quantity types shall be
            analyzed.
        
        unique_vals: ``bool``
            Passed to :meth:`self.data()`. If true, than small, random, and unique
            noise is added to the data before it is analyzed. This will effectively
            deduplicate any samples in the data (if any).

        :rtype: ``pd.DataFrame``
        
        :return: A data frame with the columns ``qtype`` (name of the quantity type),
            ``stat`` (ANOVA test statistic), ``pval``, and ``across_contexts``, where
            the latter is a semicolon-separated list of contexts the quantity type was
            compared across.
        """

        # We first have to build the data; f_oneway requires *args, where each
        # arg is a data series.
        if len(list(qtypes)) < 1 or len(list(contexts)) < 2:
            raise Exception('Requires one or quantity types and two or more contexts.')

        def anova_for_qtype(qtype: str) -> dict[str, Union[str, str, float]]:
            data_tuple = ()
            for ctx in contexts:
                data_tuple += (self.data(qtype=qtype, context=None if ctx == '__ALL__' else ctx, unique_vals=unique_vals),)
            
            stat, pval = f_oneway(*data_tuple)
            return { 'qtype': qtype, 'stat': stat, 'pval': pval, 'across_contexts': ';'.join(contexts) }

        from joblib import Parallel, delayed
        res_dicts = Parallel(n_jobs=-1)(delayed(anova_for_qtype)(qtype) for qtype in qtypes)

        return pd.DataFrame(res_dicts)
    

    def analyze_TukeyHSD(self, qtypes: Iterable[str]) -> pd.DataFrame:
        r"""
        Calculate all pairwise comparisons for the given quantity types with Tukey's
        Honest Significance Test (HSD) and return the confidence intervals. For each
        type of quantity, this method performs all of its associated contexts pair-wise
        comparisons.
        For example, given a quantity :math:`Q` and its contexts :math:`C_1,C_2,C_3`,
        this method will examine the pairs :math:`\left[\{C_1,C_2\},\{C_1,C_3\},\{C_2,C_3\}\right]`.
        For a single type of quantity, e.g., this test is useful to understand how
        different the quantity manifests across contexts. For multiple quantities, it
        also allows understanding how contexts distinguish from one another, holistically.

        qtypes: ``Iterable[str]``
            An iterable of quantity types to conduct the analysis for. For each given
            type, a separate analysis is performed and the result appended to the
            returned data frame.

        :rtype: ``pd.DataFrame``

        :return: A data frame with the columns ``group1``, ``group2``, ``meandiff``,
            ``p-adj``, ``lower``, ``upper``, and ``reject``. For details see
            :meth:`statsmodels.stats.multicomp.pairwise_tukeyhsd()`.
        """
        if len(list(qtypes)) < 1:
            raise Exception('Requires one or quantity types.')
        
        temp = self.df.copy()
        temp[self.ds['colname_context']] = '__ALL__' # Erase context
        all_data = pd.concat([temp, self.df])
        
        def tukeyHSD_for_qtype(qtype: str) -> pd.DataFrame:
            data = all_data[all_data[self.ds['colname_type']] == qtype]
            tukey = pairwise_tukeyhsd(endog=data[self.ds['colname_data']], groups=data[self.ds['colname_context']])
            temp = tukey.summary().data
            return pd.DataFrame(data=temp[1:], columns=temp[0])

        from joblib import Parallel, delayed
        res_dfs = Parallel(n_jobs=-1)(delayed(tukeyHSD_for_qtype)(qtype) for qtype in qtypes)

        return pd.concat(res_dfs)
