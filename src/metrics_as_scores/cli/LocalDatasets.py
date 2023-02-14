from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_local_datasets
from metrics_as_scores.distribution.distribution import LocalDataset


class LocalDatasetsWorkflow(Workflow):
    __doc__ = '''
This workflow lists all locally available datasets. This includes downloaded
and installed datasets, as well manually created datasets.
    '''

    def __init__(self) -> None:
        super().__init__()
    
    def _print_json_dataset(self, jsd: LocalDataset) -> None:
        self.q.print('     Name: ', style=self.style_mas, end='')
        self.q.print(jsd['name'])
        self.q.print('       ID: ', style=self.style_mas, end='')
        self.q.print(jsd['id'])

    def show_datasets(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()

        for jsd in get_local_datasets():
            self.q.print('\nDataset:')
            self.q.print(10*'-')
            self._print_json_dataset(jsd=jsd)
            self.q.print(10*'-')
        self.q.print('\n')
