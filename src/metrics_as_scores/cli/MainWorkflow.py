from pathlib import Path
from sys import path

this_dir = Path(__file__).resolve().parent
path.append(str(this_dir.parent.parent.absolute()))
from metrics_as_scores.__version__ import __version__ as mas_version

from Workflow import Workflow
from LocalWebserver import LocalWebserverWorkflow
from CreateDataset import CreateDatasetWorkflow
from KnownDatasets import KnownDatasetsWorkflow
from FitParametric import FitParametricWorkflow


class MainWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.stop = False
    
    def print_welcome(self) -> None:
        w = self.c.width
        self.c.print(w * '-')
        self.q.print(f'\n  Welcome to the Metrics-As-Scores v{mas_version} CLI!\n', style=self.style_mas)
        self.c.print(w * '-')
    
    def main_menu(self) -> Workflow:
        """
        Show the main menu of the CLI:
        """

        # The main options/Functions for M-a-S:
        res = self.askt(options=[
            ('Show Installed Datasets', 'show_local'),
            ('Show List of Known Datasets Available Online That Can Be Downloaded', 'show_known'),
            ('Download and install a known or existing dataset', 'download'),
            ('Create Own Dataset to be used with Metrics-As-Scores', 'create'),
            ('Fit Parametric Distributions for Own Dataset', 'fit'),
            ('Pre-generate distributions for usage in Web-Application', 'pre_gen'),
            ('Bundle Own dataset so it can be published', 'bundle'),
            ('Run local, interactive Web-Application using a selected dataset', 'webapp'),
            ('Quit', 'q')
        ])
        
        if res == 'show_local':
            pass
        elif res == 'show_known':
            known_ds = KnownDatasetsWorkflow()
            known_ds.show_datasets()
        elif res == 'download':
            pass
        elif res == 'create':
            create_ds = CreateDatasetWorkflow()
            create_ds.create_own()
        elif res == 'fit':
            fit_para = FitParametricWorkflow()
            fit_para.fit_parametric()
        elif res == 'bundle':
            pass
        elif res == 'webapp':
            local_server = LocalWebserverWorkflow()
            local_server.start_server()
        elif res == 'q':
            self.stop = True
            


