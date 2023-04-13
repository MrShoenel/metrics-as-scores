"""
This module contains the workflow for bundling own datasets.
"""

from os.path import commonpath
from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_local_datasets, required_files_folders_local_dataset, validate_local_dataset_files, PathStatus
from metrics_as_scores.distribution.distribution import LocalDataset
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED



class BundleDatasetWorkflow(Workflow):
    __doc__ = '''
This workflow bundles a manually created dataset into a single Zip file
that can be uploaded to, e.g., Zenodo, and registered with Metrics As
Scores in order to make it available to others as a known dataset.

In order for an own dataset to be publishable, it needs to have all
parametric fits, generated densities, references, about, etc. This
workflow will check that all requirements are fulfilled.

For an example dataset, check out https://doi.org/10.5281/zenodo.7633950
'''.strip()

    def __init__(self) -> None:
        super().__init__()
        self.use_ds: LocalDataset = None
        self.ds_dir: Path = None
    

    def _check_dirs_and_files(self) -> None:
        dirs_required, files_required = required_files_folders_local_dataset(local_ds_id=self.use_ds['id'])
        dirs_dict, files_dict = validate_local_dataset_files(dirs=dirs_required, files=files_required)

        for dir in dirs_dict.keys():
            self.print_info(text_normal='Checking directory: ', text_vital=str(dir))
            if dirs_dict[dir] == PathStatus.DOESNT_EXIST:
                raise Exception(f'Directory does not exist: {str(dir)}')
            elif dirs_dict[dir] == PathStatus.NOT_A_DIRECTORY:
                raise Exception(f'Not a directory: {str(dir)}')
        
        for file in files_dict.keys():
            self.print_info(text_normal='Checking file: ', text_vital=str(file))
            if files_dict[file] == PathStatus.DOESNT_EXIST:
                raise Exception(f'File does not exist: {str(file)}')
            if files_dict[file] == PathStatus.NOT_A_FILE:
                raise Exception(f'Not a file: {str(file)}')
    

    def _make_zip(self) -> Path:
        zip_file = self.ds_dir.joinpath('./dataset.zip')
        _, files_required = required_files_folders_local_dataset(local_ds_id=self.use_ds['id'])
        with ZipFile(file=str(zip_file), mode='x', compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for path in files_required:
                fs_path = str(path.resolve())
                base = commonpath([fs_path, str(self.ds_dir.resolve())])
                zf.write(filename=fs_path, arcname=fs_path.removeprefix(base))
        return zip_file
    

    def bundle(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()
        

        datasets = list(get_local_datasets())
        self.use_ds = self.askt(
            prompt='Select the local dataset you want to pre-generate densities for:',
            options=list([(f'{ds["name"]} [{ds["id"]}] by {", ".join(ds["author"])}', ds) for ds in datasets]))
        self.ds_dir = DATASETS_DIR.joinpath(f'./{self.use_ds["id"]}')

        try:
            self._check_dirs_and_files()
        except Exception as ex:
            self.q.print(text=str(ex), style=self.style_err)
            return
        
        
        self.q.print('')
        self.q.print(10*'-')
        self.q.print('''
We are now ready to create a Zip file. You may add additional files
to this file once it is done. Use the About.pdf file as a front matter
for your publication of this dataset (i.e., upload the Zip and the PDF
as two separate files, even though the PDF is included in the Zip).
        '''.strip())
        try:
            self.q.print('\nPacking archive (this may take a while) ...\n')
            zip_file = self._make_zip()
            self.q.print('')
            self.print_info(text_normal='Wrote archive to: ', text_vital=str(zip_file))
            self.q.print('\n' + 10*'-')
        except Exception as ex:
            self.q.print(text=str(ex), style=self.style_err)
