from typing import Callable, Iterable, Union
from nptyping import NDArray, Shape, Float, String
from src.data.metrics import MetricID
from statsmodels.distributions import ECDF as SMEcdf
from scipy.stats import kstest
from scipy.optimize import direct
import pandas as pd
import numpy as np
import scipy.stats
import pickle




class DensityFunc:
    def __init__(self, range: tuple[float, float], pdf: Callable[[float], float], cdf: Callable[[float], float], **kwargs) -> None:
        self.range = range
        self._pdf = pdf
        self._cdf = cdf
        self._practical_range: tuple[float, float] = None
        self._practical_range_pdf: tuple[float, float] = None

        self.pdf = np.vectorize(self._pdf)
        self.cdf = np.vectorize(self._min_max)


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
    def practical_range(self) -> tuple[float, float]:
        if self._practical_range is None:
            self._practical_range = self.compute_practical_range()
        return self._practical_range
    

    def compute_practical_range_pdf(self) -> tuple[float, float]:
        def obj(x):
            return -1. * np.log(1. + self.pdf(x))

        m = direct(func=obj, bounds=(self.range,))#, locally_biased=False, maxiter=15)
        return (0., 1.01 * self.pdf(m.x[0])[0])
    

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
    def __init__(self, data: NDArray[Shape["*"], Float], **kwargs) -> None:
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

        super().__init__(range=(m_lb.x, m_ub.x), pdf=pdf, cdf=cdf, kwargs=kwargs)


class KDECDF_approx(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float], resample_samples: int=200_000, compute_ranges: bool=False, **kwargs) -> None:
        # First, we'll fit an extra KDE for an approximate PDF.
        # It is used to also roughly estimate its mode.
        np.random.seed(1)
        data_pdf = data if data.shape[0] <= 10_000 else np.random.choice(a=data, size=10_000, replace=False)
        self._kde_for_pdf = gaussian_kde(dataset=data_pdf)

        self._range_data = (np.min(data), np.max(data))   
        self._kde = gaussian_kde(dataset=np.asarray(data))
        data = self._kde.resample(size=resample_samples, seed=1).reshape((-1,))
        self._ecdf = SMEcdf(x=data)

        super().__init__(range=(np.min(data), np.max(data)), pdf=self._pdf_from_kde, cdf=self._ecdf, kwargs=kwargs)

        if compute_ranges:
            self.practical_range
            self.practical_range_pdf
    
    def _pdf_from_kde(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self._kde_for_pdf.evaluate(points=np.asarray(x))



class ECDF(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float], compute_ranges: bool=False, **kwargs) -> None:
        super().__init__(range=(np.min(data), np.max(data)), pdf=gaussian_kde(dataset=data).pdf, cdf=SMEcdf(data), kwargs=kwargs)

        if compute_ranges:
            self.practical_range


class CDF(DensityFunc):
    def __init__(self, name: str, range: tuple[float, float], func: Callable[[float], float], pval: float, dstat: float, **kwargs) -> None:
        def ex(*args):
            raise Exception('PDF not supported')

        super().__init__(range=range, cdf=func, pdf=lambda _: ex, kwargs=kwargs)
        self.name = name
        self.pval = pval
        self.dstat = dstat



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
    def fit_parametric(data: NDArray[Shape["*"], Float], alpha: float=0.05, max_samples: int=10_000) -> CDF:
        distNames = ['gamma', 'gennorm', 'genexpon', 'expon', 'exponnorm',
            'exponweib', 'exponpow', 'genextreme', 'gausshyper', 'dweibull', 'invgamma', 'gilbrat','genhalflogistic', 'ncf', 'nct', 'ncx2', 'pareto', 'uniform', 'pearson3', 'mielke', 'moyal', 'nakagami', 'laplace', 'laplace_asymmetric', 'rice', 'rayleigh', 'trapezoid', 'vonmises','kappa4', 'lomax', 'loguniform', 'loglaplace', 'foldnorm', 'kstwobign', 'erlang', 'ksone','chi2', 'logistic', 'johnsonsb', 'gumbel_l', 'gumbel_r', 'genpareto', 'powerlognorm', 'bradford', 'alpha', 'tukeylambda', 'wald', 'maxwell', 'loggamma', 'fisk', 'cosine', 'burr',
            'beta', 'betaprime', 'crystalball', 'burr12', 'anglit', 'arcsine', 'gompertz', 'geninvgauss']
        
        if data.shape[0] > max_samples:
            # Then we will sub-sample to speed up the process.
            np.random.seed(1)
            data = np.random.choice(data, size=max_samples, replace=False)
        
        best_kst = None
        use_dist: tuple = None
        for distName in distNames:
            res = float('inf')

            try:
                dist = getattr(scipy.stats, distName)
                distParams = dist.fit(data)
                kst = kstest(data, cdf=dist.cdf, args=distParams)

                if kst.pvalue >= alpha and kst.statistic < res:
                    res = kst.statistic
                    best_kst = kst
                    use_dist = (distName, dist.cdf, distParams)
            except Exception as ex:
                print(ex)
                pass
        
        if use_dist is None:
            raise Exception('Cannot fit parametric distribution for given data.')
        

        def cdf(x):
            return use_dist[1](*(x, *use_dist[2]))
        
        return CDF(name=use_dist[0], range=(np.min(data), np.max(data)), func=cdf, pval=best_kst.pvalue, dstat=best_kst.statistic)
