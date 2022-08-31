from typing import Any, Callable, Iterable, Union
from typing_extensions import Self
from nptyping import NDArray, Shape, Float, String
from src.data.metrics import MetricID
from statsmodels.distributions import ECDF as SMEcdf
from scipy.stats import kstest
from scipy.optimize import direct
from scipy.stats._distn_infrastructure import rv_generic, rv_continuous
from strenum import StrEnum
import pandas as pd
import numpy as np
import scipy.stats
import pickle




class DistTransform(StrEnum):
    NONE = '<none>'
    EXPECTATION = 'E[X] (expectation)'
    MEDIAN = 'Median (50th percentile)'
    MODE = 'Mode (most likely value)'
    INFIMUM = 'Infimum (min. observed value)'
    SUPREMUM = 'Supremum (max. observed value)'



class DensityFunc:
    def __init__(self, range: tuple[float, float], pdf: Callable[[float], float], cdf: Callable[[float], float], ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, metric_id: MetricID=None, domain: str=None, **kwargs) -> None:
        self.range = range
        self._pdf = pdf
        self._cdf = cdf
        self._ideal_value = ideal_value
        self._dist_transform = dist_transform
        self._transform_value: float = None
        self._metric_id = metric_id
        self._domain = domain
        self._practical_domain: tuple[float, float] = None
        self._practical_range_pdf: tuple[float, float] = None

        self.transform_value = transform_value

        self.pdf = np.vectorize(self._pdf)
        self.cdf = np.vectorize(self._min_max)
    

    @property
    def metric_id(self) -> Union[MetricID, None]:
        return self._metric_id
    
    @property
    def domain(self) -> Union[str, None]:
        return self._domain

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
    

    def compute_practical_range(self, cutoff: float=0.995) -> tuple[float, float]:
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
            self._practical_domain = self.compute_practical_range()
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
    def load_from_file(file: str) -> 'DensityFunc':
        with open(file=file, mode='rb') as f:
            return pickle.load(file=f)



from scipy.integrate import quad
from scipy.stats import gaussian_kde

class KDECDF_integrate(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float], ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, metric_id: MetricID=None, domain: str=None, **kwargs) -> None:
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

        super().__init__(range=(m_lb.x, m_ub.x), pdf=pdf, cdf=cdf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain, kwargs=kwargs)


class KDECDF_approx(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float], resample_samples: int=200_000, compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, metric_id: MetricID=None, domain: str=None, **kwargs) -> None:
        # First, we'll fit an extra KDE for an approximate PDF.
        # It is used to also roughly estimate its mode.
        np.random.seed(1)
        data_pdf = data if data.shape[0] <= 10_000 else np.random.choice(a=data, size=10_000, replace=False)
        self._kde_for_pdf = gaussian_kde(dataset=data_pdf)

        self._range_data = (np.min(data), np.max(data))   
        self._kde = gaussian_kde(dataset=np.asarray(data))
        data = self._kde.resample(size=resample_samples, seed=1).reshape((-1,))
        self._ecdf = SMEcdf(x=data)

        super().__init__(range=(np.min(data), np.max(data)), pdf=self._pdf_from_kde, cdf=self._ecdf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain, kwargs=kwargs)

        if compute_ranges:
            self.practical_domain
            self.practical_range_pdf
    
    def _pdf_from_kde(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self._kde_for_pdf.evaluate(points=np.asarray(x))



class ECDF(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float], compute_ranges: bool=False, ideal_value: float=None, dist_transform: DistTransform=DistTransform.NONE, transform_value: float=None, metric_id: MetricID=None, domain: str=None, **kwargs) -> None:
        super().__init__(range=(np.min(data), np.max(data)), pdf=gaussian_kde(dataset=data).pdf, cdf=SMEcdf(data), ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain, kwargs=kwargs)

        if compute_ranges:
            self.practical_domain


class ParametricCDF(DensityFunc):
    def __init__(self, dist: rv_generic, pval: float, dstat: float, dist_params: tuple, range: tuple[float, float], compute_ranges: bool=False, ideal_value: float = None, dist_transform: DistTransform = DistTransform.NONE, transform_value: float = None, metric_id: MetricID = None, domain: str = None, **kwargs) -> None:
        self.dist: Union[rv_generic, rv_continuous] = dist
        self.pval = pval
        self.dstat = dstat
        self.dist_params = dist_params

        super().__init__(range=range, pdf=self.pdf, cdf=self.cdf, ideal_value=ideal_value, dist_transform=dist_transform, transform_value=transform_value, metric_id=metric_id, domain=domain, **kwargs)

        if compute_ranges:
            self.practical_domain
            self.practical_range_pdf
    
    @property
    def is_fit(self) -> bool:
        return not self.dist_params is None
    
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



class Distribution:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def available_systems(self) -> NDArray[Shape["*"], String]:
        return self.df['project'].unique()


    def get_cdf_data(self, metric_id: MetricID, systems: Iterable[str]=None, unique_vals: bool=True) -> NDArray[Shape["*"], Float]:
        new_df = self.df[self.df['metric'] == metric_id.name]
        if systems is not None:
            new_df = new_df[new_df['system'].isin(systems)]
        
        vals = new_df['value']
        if unique_vals:
            rng = np.random.default_rng(seed=1337)
            r = rng.choice(np.linspace(0, 1e-6, vals.size), vals.size, replace=False)
            # Add small but insignificant perturbations to the data to produce unique
            # values that would otherwise be eliminated by certain methods.
            vals += r
        
        return vals.to_numpy()
    
    @staticmethod
    def transform(data: NDArray[Shape["*"], Float], dist_transform: DistTransform=DistTransform.NONE) -> tuple[float, NDArray[Shape["*"], Float]]:
        if dist_transform == DistTransform.NONE:
            return (None, data)

        # Do optional transformation
        transform_value: float=None
        if dist_transform == DistTransform.EXPECTATION:
            temp = KDECDF_approx(data=data, compute_ranges=True)
            ext = temp.practical_domain[1] - temp.practical_domain[0]
            transform_value, _ = quad(func=lambda x: x * temp._pdf_from_kde(x), a=temp.practical_domain[0] - ext, b=temp.practical_domain[1] + ext, limit=250)
        elif dist_transform == DistTransform.MODE:
            temp = KDECDF_approx(data=data, compute_ranges=True)
            m = direct(func=lambda x: -1. * np.log(1. + temp._pdf_from_kde(x)), bounds=(temp.range,), locally_biased=False)
            transform_value = m.x[0] # x of where the mode is (i.e., not f(x))!
        elif dist_transform == DistTransform.MEDIAN:
            # We'll get the median from the smoothed PDF in order to also get a more smooth value
            temp = KDECDF_approx(data=data, compute_ranges=True)
            transform_value = np.median(temp._kde.resample(size=50_000, seed=2))
        elif dist_transform == DistTransform.INFIMUM:
            transform_value = np.min(data)
        elif dist_transform == DistTransform.SUPREMUM:
            transform_value = np.max(data)
        
        # Now do the convex transform: Compute the distance to the transform value!
        if transform_value is not None:
            data = np.abs(data - transform_value)
        
        return (transform_value, data)
    
    @staticmethod
    def fit_parametric(data: NDArray[Shape["*"], Float], alpha: float=0.05, max_samples: int=5_000, metric_id: MetricID=None, domain: str=None, dist_transform: DistTransform=DistTransform.NONE) -> ParametricCDF:
        distNames = ['gamma', 'gennorm', 'genexpon', 'expon', 'exponnorm',
            'exponweib', 'exponpow', 'genextreme', 'gausshyper', 'dweibull', 'invgamma', 'gilbrat','genhalflogistic', 'ncf', 'nct', 'ncx2', 'pareto', 'uniform', 'pearson3', 'mielke', 'moyal', 'nakagami', 'laplace', 'laplace_asymmetric', 'rice', 'rayleigh', 'trapezoid', 'vonmises','kappa4', 'lomax', 'loguniform', 'loglaplace', 'foldnorm', 'kstwobign', 'erlang', 'ksone','chi2', 'logistic', 'johnsonsb', 'gumbel_l', 'gumbel_r', 'genpareto', 'powerlognorm', 'bradford', 'alpha', 'tukeylambda', 'wald', 'maxwell', 'loggamma', 'fisk', 'cosine', 'burr', 'beta', 'betaprime', 'crystalball', 'burr12', 'anglit', 'arcsine', 'gompertz', 'geninvgauss']
        
        transform_value, data = Distribution.transform(data=data, dist_transform=dist_transform)
        
        if data.shape[0] > max_samples:
            # Then we will sub-sample to speed up the process.
            np.random.seed(1)
            data = np.random.choice(data, size=max_samples, replace=False)
        
        best_kst = None
        use_dist: tuple[Union[rv_generic, rv_continuous], tuple[Any]] = None
        res = float('inf')
        for dist_name in distNames:
            print(f'Trying distribution: {dist_name}')
            try:
                dist = getattr(scipy.stats, dist_name)
                dist_params = dist.fit(data)
                kst = kstest(data, cdf=dist.cdf, args=dist_params)

                if kst.pvalue >= alpha and kst.statistic < res:
                    res = kst.statistic
                    best_kst = kst
                    use_dist = (dist, dist_params)
                    break
            except Exception as ex:
                print(ex)
        
        if use_dist is None:
            raise Exception('Cannot fit parametric distribution for given data.')

        
        metrics_ideal_df = pd.read_csv('./files/metrics-ideal.csv')
        metrics_ideal_df.replace({ np.nan: None }, inplace=True)
        metrics_ideal = { x: y for (x, y) in zip(map(lambda m: MetricID[m], metrics_ideal_df.Metric), metrics_ideal_df.Ideal) }
        
        return ParametricCDF(dist=use_dist[0], pval=best_kst.pvalue, dstat=best_kst.statistic, dist_params=use_dist[1], range=(np.min(data), np.max(data)), compute_ranges=True, ideal_value=metrics_ideal[metric_id], transform_value=transform_value, dist_transform=dist_transform, metric_id=metric_id, domain=domain)
