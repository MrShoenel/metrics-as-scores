from typing import Union
from typing import Iterable
from xml.dom.minidom import Document, Element, parse
from strenum import StrEnum
from re import match


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


class QCMetricsExtractor:
    def __init__(self, file: str) -> None:
        self.xml: Document = parse(file=file)
    
    def metrics_values(self, metric_id: MetricID) -> Iterable[Union[float, int]]:
        metric: Element = self.xml.getElementById(MetricID[metric_id])
        fc: Element = metric.firstChild
        if fc.tagName == 'Value':
            # This metric has only one value, like TLOC.
            v = fc.getAttribute('value')
            if match(pattern=r'^\d+$', string=v):
                yield int(v)
            else:
                yield float(v)
        