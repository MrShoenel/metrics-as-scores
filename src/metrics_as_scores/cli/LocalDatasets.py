from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_local_datasets
from metrics_as_scores.distribution.distribution import LocalDataset


class LocalDatasetsWorkflow(Workflow):
    __doc__ = '''
This workflow lists all locally available datasets. This includes downloaded
and installed datasets, as well as manually created datasets.
    '''

    def __init__(self) -> None:
        super().__init__()
    
    def _print_json_dataset(self, jsd: LocalDataset) -> None:
        self.q.print('   Author: ', style=self.style_mas, end='')
        self.q.print(', '.join(jsd['author']))
        self.q.print('  Name/ID: ', style=self.style_mas, end='')
        self.q.print(f'{jsd["name"]} [{jsd["id"]}]')
        self.q.print('    About: ', style=self.style_mas, end='')
        self.q.print(jsd['desc'])
        self.q.print(' Features: ', style=self.style_mas, end='')
        self.q.print(', '.join(jsd['qtypes']))
        self.q.print('   Groups: ', style=self.style_mas, end='')
        self.q.print(', '.join(jsd['contexts']))

    def show_datasets(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()

        local_datasets = list(get_local_datasets())
        if len(local_datasets) == 0:
            self.q.print('\nThere are no local datasets available! You can create or download one.', style=self.style_err)

        for jsd in local_datasets:
            self.q.print('\nDataset:')
            self.q.print(10*'-')
            self._print_json_dataset(jsd=jsd)
            self.q.print(10*'-')
        self.q.print('\n')
