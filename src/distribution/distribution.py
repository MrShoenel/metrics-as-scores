from typing import Any, Callable, Iterable, Union
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
    def __init__(self, range: tuple[float, float], cdf: Callable[[float], float]) -> None:
        self.range = range
        self._practical_range: tuple[float, float] = None
        
        def func1(x):
            if x < self.range[0]:
                return 0.0
            elif x > self.range[1]:
                return 1.0
            return cdf(x)

        self.func = np.vectorize(func1)
    
    def compute_practical_range(self, cutoff: float=0.985) -> tuple[float, float]:
        def obj(x):
            return np.square(self.func(x) - cutoff)

        r = self.range
        m = direct(func=obj, bounds=(r,), f_min=0.)
        return (r[0], m.x[0])

    
    @property
    def practical_range(self) -> tuple[float, float]:
        if self._practical_range is None:
            self._practical_range = self.compute_practical_range()
        return self._practical_range
    
    def __call__(self, x: Union[float, list[float], NDArray[Shape["*"], Float]]) -> NDArray[Shape["*"], Float]:
        if np.isscalar(x) or isinstance(x, list):
            x = np.asarray(x)
        return self.func(x)
    
    def save_to_file(self, file: str) -> None:
        with open(file=file, mode='wb') as f:
            pickle.dump(obj=self, file=f)
    
    @staticmethod
    def load_from_file(file: str) -> 'DensityFunc':
        with open(file=file, mode='rb') as f:
            return pickle.load(file=f)



from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from scipy.stats import gaussian_kde

class KDECDF_integrate(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float]) -> None:
        self._kde = KernelDensity().fit(X=np.asarray(data).reshape((-1, 1)))

        def pdf(x):
            return np.exp(self._kde.score_samples(X=np.asarray(x).reshape((-1, 1))))
        
        def cdf(x):
            y, _ = quad(func=pdf, a=np.min(data), b=x)
            return y

        super().__init__(range=(np.min(data), np.max(data)), cdf=cdf)


class KDECDF_approx(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float], resample_samples: int=20_000) -> None:
        self._kde = gaussian_kde(dataset=data)
        self._ecdf = SMEcdf(x=self._kde.resample(size=resample_samples, seed=1).reshape((resample_samples,)))

        def cdf(x):
            return self._ecdf(x)

        super().__init__(range=(np.min(data), np.max(data)), cdf=cdf)



class ECDF(DensityFunc):
    def __init__(self, data: NDArray[Shape["*"], Float]) -> None:
        super().__init__(range=(np.min(data), np.max(data)), cdf=SMEcdf(data))



class CDF(DensityFunc):
    def __init__(self, name: str, range: tuple[float, float], func: Callable[[float], float], pval: float, dstat: float) -> None:
        super().__init__(range, func)
        self.name = name
        self.pval = pval
        self.dstat = dstat



class Distribution:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    

    @property
    def available_systems(self) -> NDArray[Shape["*"], String]:
        return self.df['project'].unique()


    def get_cdf_data(self, metric_id: MetricID, system: str=None, unique_vals: bool=True) -> NDArray[Shape["*"], Float]:
        new_df = self.df[self.df['metric'] == metric_id.name]
        if system is not None:
            new_df = new_df[new_df['system'] == system]
        
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
