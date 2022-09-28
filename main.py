from src.distribution.fitting import Discrete_RVs


for rv in Discrete_RVs:
    print(rv.__class__.__name__)
    for p in rv._param_info():
        print(f' - {p.__dict__}')










from copyreg import pickle
from os import walk
from typing import Iterable
from src.data.metrics import QualitasCorpusMetricsExtractor, MetricID
from pickle import dump
import pandas as pd
import numpy as np
from src.distribution.distribution import DistTransform, Dataset, Empirical, KDE_approx


d = Dataset(df=pd.read_csv('csv/metrics.csv'), attach_domain=False, attach_system=False)
anova = d.analyze_ANOVA(metric_ids=list(MetricID), domains=Dataset.domains(include_all_domain=True))
anova.to_csv('results/anova.csv', index=False)
pwrank = d.analyze_distr(metric_ids=list(MetricID))
pwrank.to_csv('results/pwrank_wtt.csv', index=False)


tukey = d.analyze_TukeyHSD(metric_ids=list(MetricID))
tukey.to_csv('results/tukey.csv', index=False)



anova = d.analyze_ANOVA(metric_ids=list(MetricID), domains=Dataset.domains(include_all_domain=True))
anova.to_csv('results/anova.csv', index=False)

with open('./results/anova.pickle', 'wb') as f:
    dump(anova, f)

#data = d.get_cdf_data(metric_id=MetricID.TLOC, unique_vals=True, systems=['aspectj', 'jre', 'jruby'])
data = d.data(metric_id=MetricID.VG, unique_vals=True)
temp = Empirical(data=data, compute_ranges=True)
print(temp.cdf(np.asarray([-1., 0., .9, 1., 1.1,])))

data = np.abs(data - np.min(data))
temp = d.fit_parametric(data, metric_id=MetricID.TLOC, dist_transform=DistTransform.EXPECTATION)
temp.save_to_file('c:/temp/bla.pickle')


dens = KDE_approx(data, compute_ranges=True)
print(dens.practical_domain)
print(dens.practical_range_pdf)
print(dens.cdf([-1.0, 0.8, 1.0, 1.5, 2.0, 20.34]))
print(dens.pdf([-1.0, 0.8, 1.0, 1.5, 2.0, 20.34]))

ecdf = Empirical(data=data)
print(ecdf([-1.0, 0.8, 1.0, 20.34]))

cdf = Dataset.fit_parametric(data=data, max_samples=1_000)
cdf.save_to_file(file='./results/cdf_VG.pickle')









bla = pd.read_csv('files/systems-domains.csv', sep=';', quotechar='"')
bla['System_QC_name'] = d.df.System.unique()
bla.to_csv('files/systems-domains2.csv')







temp = pd.read_csv('csv/__ALL__.csv')

rng = np.random.default_rng(seed=1337)
r = rng.choice(np.linspace(0, 1e-6, len(temp)), len(temp), replace=False)
#r = np.linspace(0, 1e-12, len(temp))
#np.random.seed(1337)
#np.random.shuffle(r)

nu0 = len(np.unique(temp['value']))
temp['value'] += r
nu = len(np.unique(temp['value']))


def get_file_metrics(files: list[str], proj: str, files_dir: str='./files', csv_dir: str='./csv') -> pd.DataFrame:
    dicts = list()

    for file in files:
        qcme = QualitasCorpusMetricsExtractor(file=f'{files_dir}/{file}')
        for mid in set(MetricID):
            for v in qcme.metrics_values(metric_id=mid):
                dicts.append({ 'Project': proj, 'Metric': mid.name, 'Value': v })

    df = pd.DataFrame(dicts)
    df.to_csv(f'{csv_dir}/{proj}.csv', index=False)
    return df



def convert_xml_to_csv(directory: str='./files'):
    prefixes = []
    p = None

    for _, __, files in walk(directory):
        for file in files:
            p1 = file[0:7]
            # The first 7 characters suffice to split by system,
            # yet to retain separate projects that make it up
            if p is None or p != p1:
                prefixes.append(p1)
                p = p1
                get_file_metrics(files=list(filter(lambda s: s.startswith(p1), files)), proj=p1)




def concat_csv_files(directory: str='./csv', target_file_name: str='__ALL__.csv'):
    _, __, files = list(walk(directory))[0]

    df = pd.concat(
        map(pd.read_csv, list(map(lambda s: f'{directory}/{s}', files))), ignore_index=True)

    df.to_csv(f'{directory}/{target_file_name}', index=False)