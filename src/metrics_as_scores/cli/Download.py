from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_known_datasets, get_local_datasets, KNOWN_DATASETS_FILE
from requests import get
from shutil import copyfileobj, unpack_archive
from pathlib import Path


class DownloadWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
    

    def download(self) -> None:
        self.q.print('\n' + 10*'-')
        self.q.print(f'''
This workflow access a curated list of known datasets that can be used with
Metrics As Scores. With this workflow, a known dataset can be downloaded and
installed as a local dataset. Use the workflow for listing the known datasets
and then enter the ID here.

Known datasets are loaded from: {KNOWN_DATASETS_FILE}
'''.strip())
        self.q.print('')
        self.q.print(10*'-')


        known_ds = { ds['id']: ds for ds in get_known_datasets() }
        id = self.q.text(message='Enter the ID of the dataset to download: ', validate=lambda s: s in known_ds).ask()
        local_ds = { ds['id']: ds for ds in get_local_datasets() }
        if id in local_ds:
            self.q.print(text=f'The dataset with ID "{id}" is already installed, aborting.', style = self.style_err)
            return
        
        use_ds = known_ds[id]
        dataset_dir = DATASETS_DIR.joinpath(f'./{use_ds["id"]}')
        dataset_dir.mkdir(exist_ok=False)

        self.print_info(text_normal='Downloading archive from: ', text_vital=use_ds['download'])
        zip_file = dataset_dir.joinpath('./dataset.zip')
        with get(url=use_ds['download'], stream=True) as resp:
            with open(file=str(zip_file), mode='wb') as fp:
                copyfileobj(fsrc=resp.raw, fdst=fp)
        self.q.print('Download complete. Extracting ...')

        unpack_archive(filename=str(zip_file), extract_dir=str(dataset_dir))
        self.q.print('Done! You can now use this dataset!')
        zip_file.unlink()

    

