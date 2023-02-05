from pathlib import Path
from sys import path

this_dir = Path(__file__).resolve().parent
path.append(str(this_dir.parent.parent.absolute()))
from metrics_as_scores.__version__ import __version__ as mas_version

from Workflow import Workflow
from LocalWebserver import LocalWebserverWorkflow
from CreateDataset import CreateDatasetWorkflow
from KnownDatasets import KnownDatasetsWorkflow


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
        res = self.ask(options=[
            'Show Installed Datasets',
            'Show List of Known Datasets Available Online That Can Be Downloaded',
            'Download and install a known or existing dataset',
            'Create Own Dataset to be used with Metrics-As-Scores',
            'Bundle Own dataset so it can be published',
            'Run local, interactive Web-Application using a selected dataset',
            'Quit'
        ])
        
        if res == 0:
            pass
        elif res == 1:
            known_ds = KnownDatasetsWorkflow()
            known_ds.show_datasets()
            pass
        elif res == 2:
            pass
        elif res == 3:
            create_ds = CreateDatasetWorkflow()
            create_ds.create_own()
        elif res == 4:
            pass
        elif res == 5:
            local_server = LocalWebserverWorkflow()
            local_server.start_server()
        elif res == 6:
            self.stop = True
            


