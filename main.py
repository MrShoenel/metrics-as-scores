from os import walk
from typing import Iterable
from src.data.metrics import QualitasCorpusMetricsExtractor, MetricID
import pandas as pd
import numpy as np




from src.data.metrics import MetricID
from src.distribution.distribution import Distribution, ECDF, KDECDF_approx


d = Distribution(df=pd.read_csv('csv/metrics.csv'))
data = d.get_cdf_data(metric_id=MetricID.VG, unique_vals=True)

ecdf = ECDF(data=data)
print(ecdf([-1.0, 0.8, 1.0, 20.34]))

kde = KDECDF_approx(data=d.get_cdf_data(metric_id=MetricID.VG, unique_vals=False))
print(kde.practical_range)
print(kde([-1.0, 0.8, 1.0, 1.5, 2.0, 20.34]))

cdf = Distribution.fit_parametric(data=data, max_samples=1_000)
cdf.save_to_file(file='./results/cdf_VG.pickle')














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
                dicts.append({ 'project': proj, 'metric': mid.name, 'value': v })

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