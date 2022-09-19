from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations
from typing import Any, Callable, Iterable, Union
from typing_extensions import Self
from nptyping import NDArray, Shape, Float, String
from src.data.metrics import MetricID
from statsmodels.distributions import ECDF as SMEcdf
from scipy.interpolate import interp1d
from scipy.stats import kstest, ks_2samp, f_oneway, spearmanr, ttest_ind
from scipy.optimize import direct
from scipy.stats._distn_infrastructure import rv_generic, rv_continuous
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
        rng = np.random.default_rng(seed=1)
        data_pdf = data if data.shape[0] <= 10_000 else rng.choice(a=data, size=10_000, replace=False)
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
    
    @staticmethod
    def unfitted(dist_transform: DistTransform) -> 'ParametricCDF':
        from scipy.stats._continuous_distns import norm_gen
        return ParametricCDF(dist=norm_gen(), pval=np.nan, dstat=np.nan, dist_params=None, range=(np.nan, np.nan), dist_transform=dist_transform)
    
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


class ParametricCDF_discrete(ParametricCDF):
    def pmf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        x = np.asarray(x)
        if not self.is_fit:
            return np.zeros((x.size,))
        return self.dist.pmf(*(x, *self.dist_params)).reshape((x.size,))
        
    def pdf(self, x: NDArray[Shape["*"], Float]) -> NDArray[Shape["*"], Float]:
        return self.pmf(x=x)
    
    @staticmethod
    def unfitted(dist_transform: DistTransform) -> 'ParametricCDF_discrete':
        from scipy.stats._continuous_distns import norm_gen
        return ParametricCDF_discrete(dist=norm_gen(), pval=np.nan, dstat=np.nan, dist_params=None, range=(np.nan, np.nan), dist_transform=dist_transform)



class Distribution:
    def __init__(self, df: pd.DataFrame, attach_domain: bool=False, attach_system: bool=False) -> None:
        self.df = df
        if attach_domain:
            df['domain'] = [Distribution.domain_for_system(system=system, is_qc_name=True) for system in df.system]
        if attach_system:
            df['system_org'] = [Distribution.system_qc_to_system(system_qc=system_qc) for system_qc in df.system]

    @property
    def available_systems(self) -> NDArray[Shape["*"], String]:
        return self.df['project'].unique()


    def data(self, metric_id: MetricID, domain: str=None, systems: Iterable[str]=None, unique_vals: bool=True, sub_sample: int=None) -> NDArray[Shape["*"], Float]:
        new_df = self.df[self.df['metric'] == metric_id.name]
        if domain is not None:
            new_df = new_df[new_df['domain'] == domain]
        if systems is not None:
            new_df = new_df[new_df['system'].isin(systems)]
        
        vals = new_df['value']
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
            rng = np.random.default_rng(seed=1)
            data = rng.choice(data, size=max_samples, replace=False)
        
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


    @lru_cache(maxsize=None)
    @staticmethod
    def domains(include_all_domain: bool=True) -> list[str]:
        systems_domains_df = pd.read_csv('./files/systems-domains.csv')
        domains: list[str] = systems_domains_df.Domain.unique().tolist()
        if include_all_domain:
            domains.append('__ALL__')
        return domains


    @lru_cache(maxsize=None)
    @staticmethod
    def systems_for_domain(domain: str) -> list[str]:
        systems_domains_df = pd.read_csv('./files/systems-domains.csv')
        systems_domains = dict(zip(systems_domains_df.System, systems_domains_df.Domain))
        systems_qc_names = dict(zip(systems_domains_df.System, systems_domains_df.System_QC_name))

        if domain == '__ALL__':
            return list(systems_qc_names.values())
        
        # Gather all systems with the selected domain.
        temp = filter(lambda di: di[1] == domain, systems_domains.items())
        # Map the names to the Qualitas compiled corpus:
        return list(map(lambda di: systems_qc_names[di[0]], temp))
    

    @lru_cache(maxsize=None)
    @staticmethod
    def domain_for_system(system: str, is_qc_name: bool=False) -> str:
        systems_domains_df = pd.read_csv('./files/systems-domains.csv')
        domain_dict = dict(zip(systems_domains_df[('System_QC_name' if is_qc_name else 'System')], systems_domains_df.Domain))
        return domain_dict[system]
    

    @lru_cache(maxsize=None)
    @staticmethod
    def system_qc_to_system(system_qc: str) -> str:
        systems_domains_df = pd.read_csv('./files/systems-domains.csv')
        domain_dict = dict(zip(systems_domains_df.System_QC_name, systems_domains_df.System))
        return domain_dict[system_qc]


    def analyze_ANOVA(self, metric_ids: Iterable[MetricID], domains: Iterable[str], unique_vals: bool=True) -> pd.DataFrame:
        # We first have to build the data; f_oneway requires *args, where each
        # arg is a data series.
        if len(list(metric_ids)) < 1 or len(list(domains)) < 2:
            raise Exception('Requires one or metrics and two or more domains.')

        def anova_for_metric(metric_id: MetricID) -> dict[str, Union[MetricID, str, float]]:
            data_tuple = ()
            for domain in domains:
                data_tuple += (self.data(metric_id=metric_id, systems=Distribution.systems_for_domain(domain=domain), unique_vals=unique_vals),)
            
            stat, pval = f_oneway(*data_tuple)
            return { 'metric': metric_id.name, 'stat': stat, 'pval': pval, 'across_domains': ';'.join(domains) }

        from joblib import Parallel, delayed
        res_dicts = Parallel(n_jobs=-1)(delayed(anova_for_metric)(metric_id) for metric_id in metric_ids)

        return pd.DataFrame(res_dicts)
    

    def analyze_TukeyHSD(self, metric_ids: Iterable[MetricID]) -> pd.DataFrame:
        if len(list(metric_ids)) < 1:
            raise Exception('Requires one or metrics.')
        
        temp = self.df.copy()
        temp.domain = '__ALL__' # Erase domain
        all_data = pd.concat([temp, self.df])
        
        def tukeyHSD_for_metric(metric_id: MetricID) -> pd.DataFrame:
            data = all_data[all_data.metric == metric_id.name]
            tukey = pairwise_tukeyhsd(endog=data.value, groups=data.domain)
            temp = tukey.summary().data
            return pd.DataFrame(data=temp[1:], columns=temp[0])

        from joblib import Parallel, delayed
        res_dfs = Parallel(n_jobs=-1)(delayed(tukeyHSD_for_metric)(metric_id) for metric_id in metric_ids)

        return pd.concat(res_dfs)
    

    def analyze_distr(self, metric_ids: Iterable[MetricID], use_ks_2samp: bool=True) -> pd.DataFrame:
        if len(list(metric_ids)) < 1:
            raise Exception('Requires one or metrics.')
        
        temp = self.df.copy()
        temp.domain = '__ALL__'
        all_data = pd.concat([temp, self.df])
        unique_domain_pairs: list[tuple[str, str]] = list(combinations(iterable=all_data.domain.unique(), r=2))

        # def pairwise_rank_corr(metric_id: MetricID) -> pd.DataFrame:
        #     dict_list: list[dict[str, Union[str, float]]] = [ ]

        #     for udp in unique_domain_pairs:
        #         data1 = all_data[(all_data.domain == udp[0]) & (all_data.metric == metric_id.name)].value.to_numpy()
        #         data2 = all_data[(all_data.domain == udp[1]) & (all_data.metric == metric_id.name)].value.to_numpy()

        #         l1, l2 = len(data1), len(data2)
        #         ll, ls = max(l1, l2), min(l1, l2)
        #         if ll != ls:
        #             # We need to resample the data if length not equal
        #             x_long = np.linspace(start=0., stop=1., num=ll)
        #             x_short = np.linspace(start=0., stop=1., num=ls)

        #             if l1 < l2:
        #                 data1 = interp1d(x=x_short, y=data1, kind='nearest')(x_long)
        #             else:
        #                 data2 = interp1d(x=x_short, y=data2, kind='nearest')(x_long)

        #         corr, pval = spearmanr(a=np.sort(data1), b=np.sort(data2), alternative='greater', nan_policy='raise')
        #         dict_list.append({
        #             'metric': metric_id.name, 'stat': corr, 'pval': pval, 'group1': udp[0], 'group2': udp[1]
        #         })
            
        #     return pd.DataFrame(dict_list)
        
        def compare(metric_id: MetricID) -> pd.DataFrame:
            dict_list: list[dict[str, Union[str, float]]] = [ ]

            for udp in unique_domain_pairs:
                data1 = all_data[(all_data.domain == udp[0]) & (all_data.metric == metric_id.name)].value.to_numpy()
                data2 = all_data[(all_data.domain == udp[1]) & (all_data.metric == metric_id.name)].value.to_numpy()

                stat = pval = None
                if use_ks_2samp:
                    stat, pval = ks_2samp(data1=data1, data2=data2, alternative='two-sided', method='exact')
                else:
                    stat, pval = ttest_ind(a=data1, b=data2, equal_var=False, alternative='two-sided')

                dict_list.append({
                    'metric': metric_id.name, 'stat': stat, 'pval': pval, 'group1': udp[0], 'group2': udp[1]
                })
            
            return pd.DataFrame(dict_list)

        from joblib import Parallel, delayed
        res_dfs = Parallel(n_jobs=-1)(delayed(compare)(metric_id) for metric_id in metric_ids)

        return pd.concat(res_dfs)
