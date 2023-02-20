from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_known_datasets, get_local_datasets, KNOWN_DATASETS_FILE
from shutil import unpack_archive
from wget import download
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn



class DownloadWorkflow(Workflow):
    __doc__ = f'''
This workflow access a curated list of known datasets that can be used with
Metrics As Scores. With this workflow, a known dataset can be downloaded and
installed as a local dataset. Use the workflow for listing the known datasets
and then enter the ID here.

Known datasets are loaded from: {KNOWN_DATASETS_FILE}
'''.strip()

    def __init__(self) -> None:
        super().__init__()
    

    def download(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()


        known_ds = { ds['id']: ds for ds in get_known_datasets() }
        id = self.q.text(message='Enter the ID of the dataset to download: ', validate=lambda s: s in known_ds).ask()
        local_ds = { ds['id']: ds for ds in get_local_datasets() }
        if id in local_ds:
            self.q.print(text=f'The dataset with ID "{id}" is already installed, aborting.', style = self.style_err)
            return
        
        use_ds = known_ds[id]
        dataset_dir = DATASETS_DIR.joinpath(f'./{use_ds["id"]}')
        dataset_dir.mkdir(exist_ok=False)

        self.print_info(text_normal='Downloading archive from: ', text_vital=f"{use_ds['download']}\n", arrow='\n -> ')
        zip_file = dataset_dir.joinpath('./dataset.zip')
        
        with Progress(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), BarColumn(), MofNCompleteColumn(), TextColumn("MB -"), TimeElapsedColumn(), TextColumn("-"), TimeRemainingColumn()) as progress:
            task = progress.add_task('[darkyellow]Downloading ...', total=int(round(use_ds['size'] / 1e6)))
            def update(current_bytes: int, total_bytes: int, width: int):
                progress.update(task_id=task, completed=float(current_bytes) / 1e6)
            download(url=use_ds['download'], out=str(zip_file), bar=update)

        self.q.print('Download complete. Extracting ...')

        unpack_archive(filename=str(zip_file), extract_dir=str(dataset_dir))
        self.q.print('\nDone! You can now use this dataset!\n')
        self.q.print(10*'-' + '\n')
        zip_file.unlink()

    

