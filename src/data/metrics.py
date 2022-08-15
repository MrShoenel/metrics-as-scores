from typing import Tuple, Union
from typing import Iterable
from xml.etree.cElementTree import parse, Element, iterparse
from strenum import StrEnum
from re import match
from os import walk
from typing import Iterable
import pandas as pd


class MetricID(StrEnum):
    TLOC = 'Total Lines of Code'
    NOP = 'Number of Packages'
    NOC = 'Number of Classes'
    NOI = 'Number of Interfaces'
    NOM = 'Number of Methods'
    NOF = 'Number of Attributes'
    NORM = 'Number of Overridden Methods'
    PAR = 'Number of Parameters'
    NSM = 'Number of Static Methods'
    NSF = 'Number of Static Attributes'
    WMC = 'Weighted methods per Class'
    DIT = 'Depth of Inheritance Tree'
    NSC = 'Number of Children'
    LCOM = 'Lack of Cohesion of Methods'
    MLOC = 'Method Lines of Code'
    SIX = 'Specialization Index'
    VG = 'McCabe Cyclomatic Complexity'
    NBD = 'Nested Block Depth'
    RMD = 'Normalized Distance'
    CA = 'Afferent Coupling'
    CE = 'Efferent Coupling'
    RMI = 'Instability'
    RMA = 'Abstractness'


class QualitasCorpusMetricsExtractor:
    def __init__(self, file: str) -> None:
        #self.xml: Document = parse(file=file)
        self.xml = parse(source=file).getroot()
        self.xml, self.ns = QualitasCorpusMetricsExtractor.parse_xml(file=file)
    
    @staticmethod
    def parse_xml(file: str) -> Tuple[Element, dict[str, str]]:
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
                    dicts.append({ 'system': system, 'metric': mid.name, 'value': v })

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
