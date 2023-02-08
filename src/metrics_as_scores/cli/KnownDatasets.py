from pathlib import Path
from json import load
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.distribution.distribution import KnownDataset

this_dir = Path(__file__).resolve().parent
datasets_dir = this_dir.parent.parent.parent.joinpath('./datasets')



class KnownDatasetsWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.known_datasets: list[KnownDataset] = None
        with open(file=str(datasets_dir.joinpath('./known-datasets.json')), mode='r', encoding='utf-8') as fp:
            self.known_datasets = load(fp=fp)
    
    def _print_json_dataset(self, jsd: KnownDataset) -> None:
        self.q.print('     Name: ', style=self.style_mas, end='')
        self.q.print(jsd['name'])
        self.q.print('       ID: ', style=self.style_mas, end='')
        self.q.print(jsd['id'])
        self.q.print('   Author: ', style=self.style_mas, end='')
        self.q.print(jsd['author'])
        self.q.print(' Info URL: ', style=self.style_mas, end='')
        self.q.print(jsd['info_url'])
        self.q.print(' Download: ', style=self.style_mas, end='')
        self.q.print(jsd['download'])
    
    def show_datasets(self) -> None:
        for jsd in self.known_datasets:
            self.q.print('\nDataset:')
            self.q.print(10*'-')
            self._print_json_dataset(jsd=jsd)
            self.q.print(10*'-')
        self.q.print('\n')
