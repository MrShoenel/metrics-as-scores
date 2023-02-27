"""
This module contains the main workflow (the main menu) that grants access
to all other workflows.
"""

from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.LocalWebserver import LocalWebserverWorkflow
from metrics_as_scores.cli.CreateDataset import CreateDatasetWorkflow
from metrics_as_scores.cli.KnownDatasets import KnownDatasetsWorkflow
from metrics_as_scores.cli.FitParametric import FitParametricWorkflow
from metrics_as_scores.cli.GenerateDensities import GenerateDensitiesWorkflow
from metrics_as_scores.cli.BundleOwn import BundleDatasetWorkflow
from metrics_as_scores.cli.Download import DownloadWorkflow
from metrics_as_scores.cli.LocalDatasets import LocalDatasetsWorkflow

from metrics_as_scores.__version__ import __version__ as mas_version

class MainWorkflow(Workflow):
    """
    The main workflow of the CLI is the main menu of the textual user interface.
    It provides access to all other workflows.
    """
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
        Show the main menu of the CLI.
        """

        # The main options/Functions for M-a-S:
        self.q.print('')
        res = self.askt(options=[
            ('Show Installed Datasets', 'show_local'),
            ('Show List of Known Datasets Available Online That Can Be Downloaded', 'show_known'),
            ('Download and install a known or existing dataset', 'download'),
            ('Create Own Dataset to be used with Metrics-As-Scores', 'create'),
            ('Fit Parametric Distributions for Own Dataset', 'fit'),
            ('Pre-generate distributions for usage in the Web-Application', 'pre_gen'),
            ('Bundle Own dataset so it can be published', 'bundle'),
            ('Run local, interactive Web-Application using a selected dataset', 'webapp'),
            ('Quit', 'q')
        ])
        
        if res == 'show_local':
            local_ds = LocalDatasetsWorkflow()
            local_ds.show_datasets()
        elif res == 'show_known':
            known_ds = KnownDatasetsWorkflow()
            known_ds.show_datasets()
        elif res == 'download':
            dwnld = DownloadWorkflow()
            dwnld.download()
        elif res == 'create':
            create_ds = CreateDatasetWorkflow()
            create_ds.create_own()
        elif res == 'fit':
            fit_para = FitParametricWorkflow()
            fit_para.fit_parametric()
        elif res == 'pre_gen':
            pre_gen = GenerateDensitiesWorkflow()
            pre_gen.pre_generate()
        elif res == 'bundle':
            bundler = BundleDatasetWorkflow()
            bundler.bundle()
        elif res == 'webapp':
            local_server = LocalWebserverWorkflow()
            local_server.start_server()
        elif res == 'q':
            self.stop = True
