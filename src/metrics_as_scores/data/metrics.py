from typing import Union
from typing import Iterable
from xml.etree.cElementTree import parse, Element, iterparse
from strenum import StrEnum
from re import match
from os import walk
from typing import Iterable
import pandas as pd


# Complexity metrics: A metrics suite for object oriented design
# Software package metrics: https://en.wikipedia.org/wiki/Software_package_metrics and Robert Cecil Martin (2002). Agile Software Development: Principles, Patterns and Practices. Pearson Education. ISBN 0-13-597444-5.
# Counts are low-level stuff
# In some cases, a user-defined "normality" is more useful such that new code fits right in with the existing one

class MetricID(StrEnum):
    TLOC = 'Total Lines of Code' # None -> user/context preference
    NOP = 'Number of Packages' # None
    NOC = 'Number of Classes' # None
    NOI = 'Number of Interfaces' # None
    NOM = 'Number of Methods' # None
    NOF = 'Number of Attributes' # None
    NORM = 'Number of Overridden Methods' # None
    PAR = 'Number of Parameters' # None
    NSM = 'Number of Static Methods' # None
    NSF = 'Number of Static Attributes' # None
    WMC = 'Weighted Methods per Class' # [0]; is a complexity measure; however, it depends on the definition of c, since it's \sum_{i=1}^{n} c_i, and c is deliberately left undefined; In the qualitas.class corpus, inf{c}=0, because we actually observed it
    DIT = 'Depth of Inheritance Tree' # None
    NSC = 'Number of Children' # None
    LCOM = 'Lack of Cohesion in Methods' # 0; also complexity; Zero lack would be best; however, this is also a candidate for arbitrary user-defined ideal values -> maybe you are writing a class that is a collection of static methods, like C# extensions methods, so LCOM would be high.
    MLOC = 'Method Lines of Code' # None
    SIX = 'Specialization Index' # 0; because a high specialization is undesirable; SIX = (NORM * DIT) / ([No. of Added Methods] + [No. of Inherited methods] + NORM) -> we have observed 0
    VG = 'McCabe Cyclomatic Complexity' # 1, lowest possible
    NBD = 'Nested Block Depth' # [0], or minimum possible; less nesting is considered better. In our case the average is returned and we observed values as low as ~0.3, so I guess 0 would be the theoretically lowest. However, also subject to user pref. I recommend NOT setting an ideal value since it is also a counting metric of some type.
    RMD = 'Normalized Distance' # 0; -> Distance from the main sequence (D): The perpendicular distance of a package from the idealized line A + I = 1. D is calculated as D = | A + I - 1 |. This metric is an indicator of the package's balance between abstractness and stability. A package squarely on the main sequence is optimally balanced with respect to its abstractness and stability. Ideal packages are either completely abstract and stable (I=0, A=1) or completely concrete and unstable (I=1, A=0). The range for this metric is 0 to 1, with D=0 indicating a package that is coincident with the main sequence and D=1 indicating a package that is as far from the main sequence as possible.
    CA = 'Afferent Coupling' # 0; but also user-preference
    CE = 'Efferent Coupling' # 0; but also user-preference
    RMI = 'Instability' # 0 -> Instability (I): The ratio of efferent coupling (Ce) to total coupling (Ce + Ca) such that I = Ce / (Ce + Ca). This metric is an indicator of the package's resilience to change. The range for this metric is 0 to 1, with I=0 indicating a completely stable package and I=1 indicating a completely unstable package.
    RMA = 'Abstractness' # None -> Abstractness (A): The ratio of the number of abstract classes (and interfaces) in the analyzed package to the total number of classes in the analyzed package. The range for this metric is 0 to 1, with A=0 indicating a completely concrete package and A=1 indicating a completely abstract package.


class QualitasCorpusMetricsExtractor:
    def __init__(self, file: str) -> None:
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
