from typing import Union
from typing import Iterable
from xml.etree.cElementTree import parse, Element, iterparse
from enum import Enum
from strenum import StrEnum
from re import match
from os import walk
from typing import Iterable
import pandas as pd


# Complexity metrics: A metrics suite for object oriented design
# Software package metrics: https://en.wikipedia.org/wiki/Software_package_metrics and Robert Cecil Martin (2002). Agile Software Development: Principles, Patterns and Practices. Pearson Education. ISBN 0-13-597444-5.
# Counts are low-level stuff
# In some cases, a user-defined "normality" is more useful such that new code fits right in with the existing one

temp = pd.read_csv("./files/MetricID.csv", index_col=False)
MetricID = Enum('MetricID(StrEnum)', { m: v for (m, v) in zip(temp.Metric.to_list(), temp.Value.to_list()) })
del temp


class QualitasCorpusMetricsExtractor:
    def __init__(self, file: str) -> None:
        #self.xml: Document = parse(file=file)
        self.xml = parse(source=file).getroot()
        self.xml, self.ns = QualitasCorpusMetricsExtractor.parse_xml(file=file)
    
    @staticmethod
    def parse_xml(file: str) -> tuple[Element, dict[str, str]]:
        xml_iter = iterparse(file, events=['start-ns'])
        xml_namespaces = dict(prefix_namespace_pair for _, prefix_namespace_pair in xml_iter)
        return xml_iter.root, xml_namespaces
        
    @staticmethod
    def to_numeric(value: str) -> Union[float, int]:
        if match(pattern=r'^\d+$', string=value):
            return int(value)
        return float(value)
    
    def metrics_values(self, metric_id: MetricID) -> Iterable[Union[float, int]]:
        metric = self.xml.find(f'.//*[@id="{metric_id.name}"]', self.ns)

        for elem in metric.findall(f'.//Value', self.ns):
            try:
                yield QualitasCorpusMetricsExtractor.to_numeric(value=elem.attrib['value'])
            except Exception:
                pass
    
    @staticmethod
    def get_file_metrics(files: list[str], system: str, files_dir: str='./files', csv_dir: str='./csv') -> pd.DataFrame:
        dicts = list()

        for file in files:
            qcme = QualitasCorpusMetricsExtractor(file=f'{files_dir}/{file}')
            for mid in set(MetricID):
                for v in qcme.metrics_values(metric_id=mid):
                    dicts.append({ 'System': system, 'Metric': mid.name, 'Value': v })

        df = pd.DataFrame(dicts)
        df.to_csv(f'{csv_dir}/{system}.csv', index=False)
        return df

    @staticmethod
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
                    QualitasCorpusMetricsExtractor.get_file_metrics(files=list(filter(lambda s: s.startswith(p1), files)), system=p1)

    @staticmethod
    def concat_csv_files(directory: str='./csv', target_file_name: str='__ALL__.csv'):
        _, __, files = list(walk(directory))[0]

        df = pd.concat(
            map(pd.read_csv, list(map(lambda s: f'{directory}/{s}', files))), ignore_index=True)

        df.to_csv(f'{directory}/{target_file_name}', index=False)
