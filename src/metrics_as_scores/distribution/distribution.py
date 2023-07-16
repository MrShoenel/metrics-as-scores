"""
This module contains the base class for all densities as used in the web
application, as well as all of its concrete implementations. Also, it
contains enumerations and typings that describe datasets.
"""

import pandas as pd
import numpy as np
from abc import ABC
from typing import Callable, Iterable, Literal, Union, TypedDict
from typing_extensions import Self
from nptyping import NDArray, Shape, Float
from itertools import combinations
from metrics_as_scores.distribution.fitting import StatisticalTest
from metrics_as_scores.tools.funcs import cdf_to_ppf
from metrics_as_scores.distribution.fitting import StatTest_Types
from statsmodels.distributions import ECDF as SMEcdf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import gaussian_kde, f_oneway, mode, kruskal
from scipy.integrate import quad
from scipy.optimize import direct
from scipy.stats import ks_2samp, ttest_ind
from scipy.stats._distn_infrastructure import rv_generic, rv_continuous
from joblib import Parallel, delayed
from tqdm import tqdm
from strenum import StrEnum



class DistTransform(StrEnum):
    """
    This is an enumeration of transforms applicable to distributions of a quantity.
    A transform first computes the desired ideal (transform) value from the given
    density (e.g., the expectation) and then transforms the initial distribution of
    values into a distribution of distances.
    """
    NONE = '<none>'
    """Do not apply any transform."""

    EXPECTATION = 'E[X] (expectation)'
    r"""
    Compute the expectation of the random variable.
    This is similar to :math:`\mathbb{E}[X]=\int_{-\infty}^{\infty}x*f_X(x) dx` for a
    continuous random variable.
    """

    MEDIAN = 'Median (50th percentile)'
    """
    Compute the median (50th percentile) of the random variable. The median is defined
    as the value that splits a probability distribution into a lower and higher half.
    """
    
    MODE = 'Mode (most likely value)'
    """
    The mode of a random variable is the most frequently occurring value, i.e., the
    value with the highest probability (density).
    """

    INFIMUM = 'Infimum (min. observed value)'
    """
    The infimum is the lowest observed value of some empirical random variable.
    """

    SUPREMUM = 'Supremum (max. observed value)'
    """
    The supremum is the highest observed value of some empirical random variable.
    """


class JsonDataset(TypedDict):
    """
    This class is the base class for the :py:class:`LocalDataset` and the
    :py:class:`KnownDataset`. Each manifest should have a name, id, description,
    and author.
    """
    name: str
    desc: str
    id: str
    author: list[str]


class LocalDataset(JsonDataset):
    """
    This dataset extends the :py:class:`JsonDataset` and adds properties that
    are filled out when locally creating a new dataset.
    """
    origin: str
    colname_data: str
    colname_type: str
    colname_context: str
    qtypes: dict[str, Literal['continuous', 'discrete']]
    desc_qtypes: dict[str, str]
    contexts: list[str]
    desc_contexts: dict[str, str]
    ideal_values: dict[str, Union[int, float]]


class KnownDataset(JsonDataset):
    """
    This dataset extends the :py:class:`JsonDataset` with properties that are
    known about datasets that are available to Metrics As Scores online.
    """
    info_url: str
    download: str
    size: int
    size_extracted: int


class Density(ABC):
    """
    This is the abstract base class for parametric and empirical densities. A
    :py:class:`Density` represents a concrete instance of some random variable
    and its PDF, CDF, and PPF. It also stores information about this concrete
    instance came to be (e.g., by some concrete transform).

    This class provides a set of common getters and setters and also provides
    some often needed conveniences, such as computing the practical domain.
    As for the PDF, CDF, and PPF, all known sub-classes have a specific way of
    obtaining these, and this class' responsibility lies in vectorizing these
    functions.
    """

    def __init__(
        self,
        range: tuple[float, float],
        pdf: Callable[[float], float],
        cdf: Callable[[float], float],
        ppf: Callable[[float], float]=None,
        ideal_value: float=None,
        dist_transform: DistTransform=DistTransform.NONE,
        transform_value: float=None,
        qtype: str=None,
        context: str=None,
        **kwargs
    ) -> None:
        """
        range: ``tuple[float, float]``
            The range of the data.
        
        pdf: ``Callable[[float], float]``
            The probability density function.
        
        cdf: ``Callable[[float], float]``
            The cumulative distribution function.
        
        ppf: ``Callable[[float], float]``
            The percent point (quantile) function.
        
        ideal_value: ``float``
            Some quantities have an ideal value. It can be provided here.
        
        dist_transform: ``DistTransform``
            The data transform that was applied while obtaining this density.
        
        transform_value: ``float``
            Optional transform value that was applied during transformation.
        
        qtype: ``str``
            The type of quantity for this density.
        
        context: ``str``
            The context of this quantity.
        """
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
        """Getter for the quantity type."""
        return self._qtype
    
    @property
    def context(self) -> Union[str, None]:
        """Getter for the context."""
        return self._context

    @property
    def ideal_value(self) -> Union[float, None]:
        """Getter for the ideal value (if any)."""
        return self._ideal_value

    @property
    def dist_transform(self) -> DistTransform:
        """Getter for the data transformation."""
        return self._dist_transform

    @property
    def transform_value(self) -> Union[float, None]:
        """Getter for the used transformation value (if any)."""
        return self._transform_value
    
    @transform_value.setter
    def transform_value(self, value: Union[float, None]) -> Self:
        """Setter for the used transformation value."""
        self._transform_value = value
        return self


    def _min_max(self, x: float) -> float:
        """
        Used to safely vectorize a CDF, such that it returns `0.0` for when
        `x` lies before our range, and `1.0` if `x` lies beyond our range.

        x: ``float``
            The `x` to obtain the CDF's `y` for.
        
        :return:
            A value in the range :math:`[0,1]`.
        """
        return np.clip(a=self._cdf(x), a_min=0.0, a_max=1.0)
    

    def compute_practical_domain(self, cutoff: float=0.995) -> tuple[float, float]:
        """
        It is quite common that domains extend into distant regions to accommodate
        even the farthest outliers. This is often counter-productive, especially in
        the web application. There, we often want to show most of the distribution,
        so we compute a practical range that cuts off the most extreme outliers.
        This is useful to showing some default window.

        cutoff: ``float``
            The percentage of values to include. The CDF is optimized to find some `x`
            for which it peaks at the cutoff. For the lower bound, we subtract from
            CDF the cutoff.
        
        :rtype: tuple[float, float]

        :return:
            The practical domain, cut off for both directions.
        """
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
        """
        Getter for the practical domain. This is a lazy getter that only
        computes the practical domain if it was not done before.
        """
        if self._practical_domain is None:
            self._practical_domain = self.compute_practical_domain()
        return self._practical_domain
    

    def compute_practical_range_pdf(self) -> tuple[float, float]:
        """
        Similar to :py:meth:compute_practical_domain(), this method computes a practical
        range for the PDF. This method determines the location of the PDF's highest mode.

        :return:
            Returns a tuple where the first element is always `0.0` and the second is
            the `y` of the highest mode (i.e., returns the `y` of the mode, not `x`, its
            location).
        """
        def obj(x):
            return -1. * np.log(1. + self.pdf(x))

        m = direct(func=obj, bounds=(self.range,), locally_biased=False)#, maxiter=15)
        return (0., self.pdf(m.x[0])[0])
    

    @property
    def practical_range_pdf(self) -> tuple[float, float]:
        """
        Lazy getter for the practical range of the PDF.
        """
        if self._practical_range_pdf is None:
            self._practical_range_pdf = self.compute_practical_range_pdf()
        return self._practical_range_pdf

    
    def __call__(self, x: Union[float, list[float], NDArray[Shape["*"], Float]]) -> NDArray[Shape["*"], Float]:
        """
        Allow objects of type :py:class:`Density` to be callable. Calls the vectorized
        CDF under the hood.
        """
        if np.isscalar(x) or isinstance(x, list):
            x = np.asarray(x)
        return self.cdf(x)


class KDE_integrate(Density):
    r"""
    The purpose of this class is to use an empirical (typically Gaussian) PDF and to
    also provide a smooth CDF that is obtained by integrating the PDF:
    :math:`F_X(x)=\int_{-\infty}^{x} f_X(t) dt`.
    While this kind of CDF is smooth and precise, evaluating it is obviously slow.
    Therefore, :py:class:`KDE_approx` is used in practice.
    """
    def __init__(self, data: NDArray[Shape["*"], Float], ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self._data = data
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

        super().__init__(range=(m_lb.x, m_ub.x), pdf=pdf, cdf=cdf, ppf=None, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, kwargs=kwargs)
    
    def init_ppf(self, cdf_samples: int=100) -> 'KDE_integrate':
        """
        Initializes the PPF. We get `x` and `y` from the CDF. Then, we swap the two
        and interpolate a PPF. Since obtaining each `y` from the CDF means we need
        to compute an integral, be careful with setting a high number of ``cdf_samples``.

        cdf_samples: ``int``
            The number of samples to take from the CDF (which is computed by integrating the
            PDF, so be careful).
        """
        self._ppf = cdf_to_ppf(cdf=self.cdf, x=self._data, y_left=np.min(self._data), y_right=np.max(self._data), cdf_samples=cdf_samples)
        self.ppf = np.vectorize(self._ppf)
        return self


class KDE_approx(Density):
    """
    This kind of density uses Kernel Density Estimation to obtain a PDF, and an empirical
    CDF (ECDF) to provide a cumulative distribution function. The advantage is that both,
    PDF and CDF, are fast.
    The PPF is the inverted and interpolated CDF, so it is fast, too. The data used for the
    PDF is limited to 10_000 samples using deterministic sampling without replacement. The
    used for CDF is obtained by sampling a large number (typically 200_000) of data points
    from the Gaussian KDE, in order to make it smooth.
    """
    def __init__(self, data: NDArray[Shape["*"], Float], resample_samples: int=200_000, compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        """
        For the other parameters, please refer to :py:meth:`Density.__init__()`.

        resample_samples: ``int``
            The amount of samples to take from the Gaussian KDE. These samples are then used
            to estimate an as-smooth-as-possible CDF (and PPF thereof).
        
        compute_ranges: ``bool``
            Whether or not to compute the practical domain of the data and the practical range
            of the PDF. Both of these use optimization to find the results.
        """
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
        """
        Shortcut getter for the jittered, two-sample KS-test's p-value.
        """
        return self.stat_test.tests['ks_2samp_jittered']['pval']
    
    @property
    def stat(self) -> float:
        """
        Shortcut getter for the jittered, two-sample KS-test's test statistic (D-value).
        """
        return self.stat_test.tests['ks_2samp_jittered']['stat']


class Empirical(Density):
    """
    This kind of density does not apply any smoothing for CDF, but rather uses a
    straightforward ECDF for the data as given. The PDF is determined using Gaussian
    KDE.
    """
    def __init__(self, data: NDArray[Shape["*"], Float], compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        """
        For the other parameters, please refer to :py:meth:`Density.__init__()`.

        compute_ranges: ``bool``
            Whether or not to compute the practical domain of the data and the practical range
            of the PDF. Both of these use optimization to find the results.
        """
        self._ecdf = SMEcdf(data)
        self._ppf_interp = cdf_to_ppf(cdf=np.vectorize(self._ecdf), x=data, y_left=np.min(data), y_right=np.max(data))

        super().__init__(range=(np.min(data), np.max(data)), pdf=gaussian_kde(dataset=data).pdf, cdf=self._ecdf, ppf=self._ppf_from_ecdf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, qtype=qtype, context=context, kwargs=kwargs)

        if compute_ranges:
            self.practical_domain
    
    def _ppf_from_ecdf(self, q: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self._ppf_interp(q)


class Empirical_discrete(Empirical):
    """
    Inherits from :py:class:`Empirical` and is used when the underlying quantity is
    discrete and not continuous. As PDF, this function uses a PMF that is determined
    by the frequencies of each discrete datum.
    """
    def __init__(self, data: NDArray[Shape["*"], Float], ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        self._data_valid = not (data.shape[0] == 1 and np.isnan(data[0]))
        data_int = np.full(shape=data.shape, fill_value=np.nan) if not self._data_valid else np.rint(data).astype(int)
        if self._data_valid and not np.allclose(a=data, b=data_int, rtol=1e-10, atol=1e-12):
            raise Exception('The data does not appear to be integral.')

        self._unique, self._counts = np.unique(data_int, return_counts=True)
        self._unique, self._counts = np.full(shape=self._unique.shape, fill_value=np.nan) if not self._data_valid else self._unique.astype(int), self._counts.astype(int)
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
        """Returns `True` if the given data is valid."""
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
        """
        Used to return an explicit unfit instance of :py:class:`Empirical_discrete`.
        This is used when, for example, continuous (real) data is given to the
        constructor. We still need an instance of this density in the web application
        to show an error (e.g., that there are no discrete empirical densities for
        continuous data).
        """
        return Empirical_discrete(data=np.asarray([np.nan]), dist_transform=dist_transform)


class Parametric(Density):
    """
    This density encapsulates a parameterized and previously fitted random variable.
    Random variables in :py:mod:`scipy.stats` come with PDF/PMF, CDF, PPF, etc. so
    we just use these and forward calls to them.
    """
    def __init__(self, dist: rv_generic, dist_params: tuple, range: tuple[float, float], stat_tests: dict[str, float], use_stat_test: StatTest_Types='ks_2samp_jittered', compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, qtype: str=None, context: str=None, **kwargs) -> None:
        """
        For the other parameters, please refer to :py:meth:`Density.__init__()`.

        dist: ``rv_generic``
            An instance of the random variable to use.
        
        dist_params: ``tuple``
            A tuple of parameters for the random variable. The order of the parameters
            is important since it is not a dictionary.
        
        stat_tests: ``dict[str, float]``
            A (flattened) dictionary of previously conducted statistical tests. This is
            used later to choose some best-fitting parametric density by a specific test.
        
        use_stat_test: ``StatTest_Types``
            The name of the chosen statistical test used to determine the goodnes of fit.
        
        compute_ranges: ``bool``
            Whether or not to compute the practical domain of the data and the practical range
            of the PDF. Both of these use optimization to find the results.
        """
        if not isinstance(stat_tests, dict) or isinstance(stat_tests, StatisticalTest):
            raise Exception(f'This class requires a dictionary of statistical tests, not an instance of {StatisticalTest.__name__}.')
        for (key, val) in stat_tests.items():
            if not key.endswith('_pval') and not key.endswith('_stat'):
                raise Exception(f'Key not allowed: "{key}".')
            if not isinstance(val, float):
                raise Exception(f'Value for key "{key}" is not numeric.')

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
        """
        Used to return an explicit unfit instance of :py:class:`Parametric`. This is
        used in case when not a single maximum likelihood fit was successful for a
        number of random variables. We still need an instance of this density in the
        web application to show an error (e.g., that it was not possible to fit any
        random variable to the selected quantity).
        """
        from scipy.stats._continuous_distns import norm_gen
        return Parametric(dist=norm_gen(), dist_params=None, range=(np.nan, np.nan), stat_tests={}, dist_transform=dist_transform)

    @property
    def use_stat_test(self) -> StatTest_Types:
        """Getter for the selected statistical test."""
        return self._use_stat_test
    
    @use_stat_test.setter
    def use_stat_test(self, st: StatTest_Types) -> Self:
        """Setter for the type of statistical test to use."""
        self._use_stat_test = st
        return self
    
    @property
    def pval(self) -> float:
        """Shortcut getter for the p-value of the selected statistical test."""
        if not self.is_fit:
            raise Exception('Cannot return p-value for non-fitted random variable.')
        return self.stat_tests[f'{self.use_stat_test}_pval']
    
    @property
    def stat(self) -> float:
        """Shortcut getter for the test statistic of the selected statistical test."""
        if not self.is_fit:
            raise Exception('Cannot return statistical test statistic for non-fitted random variable.')
        return self.stat_tests[f'{self.use_stat_test}_stat']
    
    @property
    def is_fit(self) -> bool:
        """Returns `True` if this instance is not an explicitly unfit instance."""
        return not self.dist_params is None and not np.any(np.isnan(self.dist_params))
    
    @property
    def practical_domain(self) -> tuple[float, float]:
        """Overridden to return a practical domain of :math:`[0,0]` in case this instance is unfit."""
        if not self.is_fit:
            return (0., 0.)
        return super().practical_domain
    
    @property
    def practical_range_pdf(self) -> tuple[float, float]:
        """Overridden to return a practical PDF range of :math:`[0,0]` in case this instance is unfit."""
        if not self.is_fit:
            return (0., 0.)
        return super().practical_range_pdf
    
    @property
    def dist_name(self) -> str:
        """Shortcut getter for the this density's random variable's class' name."""
        return self.dist.__class__.__name__
    
    def pdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        """
        Overridden to call the encapsulated distribution's PDF. If this density is unfit,
        always returns an array of zeros of same shape as the input.
        """
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.pdf(*(x, *self.dist_params)).reshape((x.size,))
    
    def cdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        """
        Overridden to call the encapsulated distribution's CDF. If this density is unfit,
        always returns an array of zeros of same shape as the input.
        """
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.cdf(*(x, *self.dist_params)).reshape((x.size,))
    
    def ppf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        """
        Overridden to call the encapsulated distribution's PPF. If this density is unfit,
        always returns an array of zeros of same shape as the input.
        """
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.ppf(*(x, *self.dist_params)).reshape((x.size,))
    
    def compute_practical_domain(self, cutoff: float=0.9985) -> tuple[float, float]:
        """
        Overridden to exploit having available a PPF of a fitted random variable.
        It can be used to find the practical domain instantaneously instead of
        having to solve an optimization problem.

        cutoff: ``float``
            The percentage of values to include. The CDF is optimized to find some `x`
            for which it peaks at the cutoff. For the lower bound, we subtract from
            CDF the cutoff. Note that the default value for the cutoff was adjusted
            here to extend a little beyond what is good for other types of densities.
        
        :rtype: tuple[float, float]

        :return:
            The practical domain, cut off for both directions. If this random variable
            is unfit, returns :py:class:`Density`'s :py:meth:`compute_practical_domain()`.
        """
        if not self.is_fit:
            return self.range
        
        return (self.ppf(1.0 - cutoff)[0], self.ppf(cutoff)[0])


class Parametric_discrete(Parametric):
    """
    This type of density inherits from :py:class:`Parametric` and is its counterpart
    for discrete (integral) data. It adds an explicit function for the probability mass
    and makes the inherited PDF return the PMF's result.
    """

    def pmf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        """
        Implemented to call the encapsulated distribution's PMF. If this density is unfit,
        always returns an array of zeros of same shape as the input.
        """
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.pmf(*(x, *self.dist_params)).reshape((x.size,))
        
    def pdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        """
        Overridden to return the result of the :py:meth:`pmf()`. Note that in any case,
        a density's function :py:meth:`pdf()` is called (i.e., the callers never call
        the PMF). Therefore, it is easier catch these calls and redirect them to the PMF.
        """
        return self.pmf(x=x)
    
    @staticmethod
    def unfitted(dist_transform: DistTransform) -> 'Parametric_discrete':
        """
        Used to return an explicit unfit instance of :py:class:`Parametric`. This is
        used in case when not a single maximum likelihood fit was successful for a
        number of random variables. We still need an instance of this density in the
        web application to show an error (e.g., that it was not possible to fit any
        random variable to the selected quantity).
        """
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
        """Shortcut getter for the manifest's quantity types."""
        return list(self.ds['qtypes'].keys())

    def contexts(self, include_all_contexts: bool=False) -> Iterable[str]:
        """
        Returns the manifest's defined contexts as a generator. Sometimes we need to ignore
        the context and aggregate a quantity type across all defined contexts. Then, a virtual
        context called `__ALL__` is used.

        include_all_contexts: ``bool``
            Whether to also yield the virtual `__ALL__`-context.
        """
        yield from self.ds['contexts']
        if include_all_contexts:
            yield '__ALL__'
    
    @property
    def ideal_values(self) -> dict[str, Union[float, int, None]]:
        """Shortcut getter for the manifest's ideal values."""
        return self.ds['ideal_values']
    
    def is_qtype_discrete(self, qtype: str) -> bool:
        """
        Returns whether a given quantity type is discrete.
        """
        return self.ds['qtypes'][qtype] == 'discrete'
    
    def qytpe_desc(self, qtype: str) -> str:
        """
        Returns the description associated with a quantity type.
        """
        return self.ds['desc_qtypes'][qtype]
    
    def context_desc(self, context: str) -> Union[str, None]:
        """
        Returns the description associated with a context (if any).
        """
        return self.ds['desc_contexts'][context]
    
    @property
    def quantity_types_continuous(self) -> list[str]:
        """
        Returns a list of quantity types that are continuous (real-valued).
        """
        return list(filter(lambda qtype: not self.is_qtype_discrete(qtype=qtype), self.quantity_types))

    @property
    def quantity_types_discrete(self) -> list[str]:
        """
        Returns a list of quantity types that are discrete (integer-valued).
        """
        return list(filter(lambda qtype: self.is_qtype_discrete(qtype=qtype), self.quantity_types))

    def data(self, qtype: str, context: Union[str, None, Literal['__ALL__']]=None, unique_vals: bool=True, sub_sample: int=None) -> NDArray[Shape["*"], Float]:
        """
        This method is used to select a subset of the data, that is specific to at least
        a type of quantity, and optionally to a context, too.

        qtype: ``str``
            The name of the quantity type to get data for.

        context: ``Union[str, None, Literal['__ALL__']]``
            You may specify a context to further filter the data by. Data is always specific
            to a quantity type, and sometimes to a context. If not context-based filtering is
            desired, pass ``None`` or ``__ALL__``.
        
        unique_vals: ``bool``
            If `True`, some small jitter will be added to the data in order to make it unique.
        
        sub_sample: ``int``
            Optional unsigned integer with number of samples to take in case the dataset is
            very large. It is only applied if the number is smaller than the data's size.
        """
        new_df = self.df[self.df[self.ds['colname_type']] == qtype]
        if context is not None and context != '__ALL__':
            # Check if context exists:
            if not context in list(self.contexts(include_all_contexts=False)):
                raise Exception(f'The context "{context}" is not known.')
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
    
    def num_observations(self) -> Iterable[tuple[str, str, int]]:
        """
        Returns the number of observations for each quantity type in each context.

        :rtype: ``Iterable[tuple[str, str, int]]``
            The first element is the context, the second the quantity type, and the
            third is the number of observations.

        :return: Returns an iterable generator.
        """
        for ctx in self.contexts(include_all_contexts=False):
            for qtype in self.quantity_types:
                yield (ctx, qtype, self.data(qtype=qtype, context=ctx, unique_vals=False).shape[0])
    
    def has_sufficient_observations(self, raise_if_not: bool=True) -> bool:
        """
        Helper method to check whether each quantity type in each context has at least
        two observations.

        raise_if_not: ``bool``
            If set to `True`, will raise an exception instead of returning `False` in
            case there are insufficiently many observations. The exception is more
            informative as it includes the the context and quantity type.
        
        :rtype: ``bool``

        :return: A boolean indicating whether this Dataset has sufficiently many observations
            for each and every quantity type.
        """
        for ctx, qtype, num_obs in self.num_observations():
            if num_obs < 2:
                if raise_if_not:
                    raise Exception(f'The quantity type "{qtype}" in context "{ctx}" has insufficient ({num_obs}) observation(s).')
                else:
                    return False
        return True
    
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
    

    def analyze_groups(self, use: Literal['anova', 'kruskal'], qtypes: Iterable[str], contexts: Iterable[str], unique_vals: bool=True) -> pd.DataFrame:
        """
        For each given type of quantity, this method performs an ANOVA across all
        given contexts.

        use: ``Literal['anova', 'kruskal']``
            Indicates which method for comparing groups to use. We can either conduct
            an ANOVA or a Kruskal-Wallis test.

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
        use_method: Callable = None
        if use == 'anova':
            use_method = f_oneway
        elif use == 'kruskal':
            use_method = kruskal
        else:
            raise Exception(f'Method "{use}" is not supported.')
        

        # We first have to build the data; f_oneway requires *args, where each
        # arg is a data series.
        if len(list(qtypes)) < 1 or len(list(contexts)) < 2:
            raise Exception('Requires one or quantity types and two or more contexts.')

        def anova_for_qtype(qtype: str) -> dict[str, Union[str, str, float]]:
            samples = ()
            for ctx in contexts:
                samples += (self.data(qtype=qtype, context=None if ctx == '__ALL__' else ctx, unique_vals=unique_vals),)
            
            stat, pval = use_method(*samples)
            return { 'qtype': qtype, 'stat': stat, 'pval': pval, 'across_contexts': ';'.join(contexts) }

        res_dicts = Parallel(n_jobs=-1)(delayed(anova_for_qtype)(qtype) for qtype in tqdm(qtypes))

        return pd.DataFrame(res_dicts)
    

    def analyze_TukeyHSD(self, qtypes: Iterable[str]) -> pd.DataFrame:
        r"""
        Calculate all pairwise comparisons for the given quantity types with Tukey's
        Honest Significance Test (HSD) and return the confidence intervals. For each
        type of quantity, this method performs all of its associated contexts pairwise
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

        res_dfs = Parallel(n_jobs=-1)(delayed(tukeyHSD_for_qtype)(qtype) for qtype in tqdm(qtypes))

        return pd.concat(res_dfs)
    

    def analyze_distr(self, qtypes: Iterable[str], use_ks_2samp: bool=True, ks2_max_samples=40_000) -> pd.DataFrame:
        """
        Performs the two-sample Kolmogorov--Smirnov test or Welch's t-test for two or
        more types of quantity. Performs the test for all unique pairs of quantity types.

        qtypes: ``Iterable[str]``
            An iterable of quantity types to test in a pair-wise manner.
        
        use_ks_2samp: ``bool``
            If `True`, use the two-sample Kolmogorov--Smirnov; Welch's t-test, otherwise.
        
        ks2_max_samples: ``int``
            Unsigned integer used to limit the number of samples used in KS2-test. For larger
            numbers than the default, it may not be possible to compute it exactly.

        :rtype: ``pd.DataFrame``

        :return:
            A data frame with columns `qtype`, `stat`, `pval`, `group1`, and `group2`.
        """
        if len(list(qtypes)) < 1:
            raise Exception('Requires one or more quantity types.')
        
        unique_context_pairs: list[tuple[str, str]] = list(combinations(iterable=self.contexts(include_all_contexts=True), r=2))
        
        def compare(qtype: str) -> pd.DataFrame:
            dict_list: list[dict[str, Union[str, float]]] = [ ]

            for udp in unique_context_pairs:
                data1 = self.data(qtype=qtype, context=udp[0])
                data2 = self.data(qtype=qtype, context=udp[1])

                stat = pval = None
                if use_ks_2samp:
                    rng = np.random.default_rng(seed=1)
                    data1 = data1 if data1.shape[0] <= ks2_max_samples else rng.choice(a=data1, size=ks2_max_samples, replace=False)
                    data2 = data2 if data2.shape[0] <= ks2_max_samples else rng.choice(a=data2, size=ks2_max_samples, replace=False)

                    stat, pval = ks_2samp(data1=data1, data2=data2, alternative='two-sided', method='exact')
                else:
                    stat, pval = ttest_ind(a=data1, b=data2, equal_var=False, alternative='two-sided')

                dict_list.append({
                    'qtype': qtype, 'stat': stat, 'pval': pval, 'group1': udp[0], 'group2': udp[1]
                })
            
            return pd.DataFrame(dict_list)

        res_dfs = Parallel(n_jobs=-1)(delayed(compare)(qtype) for qtype in tqdm(self.quantity_types))
        return pd.concat(res_dfs)
