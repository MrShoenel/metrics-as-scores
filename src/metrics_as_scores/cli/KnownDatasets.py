"""
This module contains the workflow for listing known datasets that are
available online and may be downloaded.
"""

from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_known_datasets, format_file_size, KNOWN_DATASETS_FILE
from metrics_as_scores.distribution.distribution import KnownDataset


class KnownDatasetsWorkflow(Workflow):
    __doc__ = f'''
This workflow access a curated online list of known datasets that were
designed to work with Metrics As Scores. The list accessed is at:

{KNOWN_DATASETS_FILE}

If you would like to have your own dataset added to this list, open an
issue in the Github repository of Metrics As Scores (also, check out the
contributing guidelines).
'''.strip()
    def __init__(self) -> None:
        super().__init__()
        self.q.print('\nFetching available datasets ...\n')
        self.known_datasets: list[KnownDataset] = None
    
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
        self.q.print('     Size: ', style=self.style_mas, end='')
        self.q.print(f'{format_file_size(jsd["size"])} ({format_file_size(jsd["size_extracted"])} extracted)')
    
    def show_datasets(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()

        self._wait_for(what_for='to fetch and list available datasets')
        self.known_datasets = get_known_datasets(use_local_file=False)

        for jsd in self.known_datasets:
            self.q.print('\nDataset:')
            self.q.print(10*'-')
            self._print_json_dataset(jsd=jsd)
            self.q.print(10*'-')
        self.q.print('\n')
