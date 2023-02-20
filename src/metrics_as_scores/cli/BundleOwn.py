import pandas as pd
from os.path import commonpath
from metrics_as_scores.__init__ import DATASETS_DIR
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import get_local_datasets
from metrics_as_scores.distribution.distribution import DistTransform, Parametric, Parametric_discrete, Empirical, Empirical_discrete, KDE_approx, Dataset, LocalDataset
from itertools import product
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
        self.ds: Dataset = None
        self.ds_dir: Path = None

        self.dirs_required: list[Path] = None
        self.files_required: list[Path] = None
    

    def _check_files(self) -> None:
        for d in self.dirs_required:
            self.print_info(text_normal='Checking directory: ', text_vital=str(d))
            if not d.exists():
                raise Exception(f'Directory does not exist: {str(d)}')
            elif not d.is_dir():
                raise Exception(f'Not a directory: {str(d)}')
        
        for f in self.files_required:
            self.print_info(text_normal='Checking file: ', text_vital=str(f))
            if not f.exists():
                raise Exception(f'File does not exist: {str(f)}')
            elif not f.is_file():
                raise Exception(f'Not a file: {str(f)}')
    

    def _make_zip(self) -> Path:
        zip_file = self.ds_dir.joinpath('./dataset.zip')
        with ZipFile(file=str(zip_file), mode='x', compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for path in self.files_required:
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
        self.ds = Dataset(ds=self.use_ds, df=pd.read_csv(str(self.ds_dir.joinpath('./org-data.csv'))))

        dir_densities = self.ds_dir.joinpath(f'./densities')
        dir_fits = self.ds_dir.joinpath(f'./fits')
        dir_stests = self.ds_dir.joinpath(f'./stat-tests')
        dir_web = self.ds_dir.joinpath(f'./web')

        self.dirs_required = [
            dir_densities,
            dir_fits,
            dir_stests,
            dir_web
        ]

        self.files_required = list([
            dir_densities.joinpath(f'./densities_{perm[0].__name__}_{perm[1].name}.pickle')
            for perm in product([
                Parametric, Parametric_discrete, Empirical, Empirical_discrete, KDE_approx
            ], list(DistTransform))
        ]) + list([
            dir_fits.joinpath(f'./pregen_distns_{dt.name}.pickle') for dt in list(DistTransform)
        ]) + list([
            dir_stests.joinpath(f'./{file}.csv') for file in ['anova', 'ks2samp', 'tukeyhsd']
        ]) + list([
            dir_web.joinpath(f'./{file}.html') for file in ['about', 'references']
        ]) + list([
            self.ds_dir.joinpath(f'./{file}') for file in [
                'About.pdf',
                'manifest.json',
                'org-data.csv',
                'refs.bib'
            ]
        ])

        try:
            self._check_files()
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
