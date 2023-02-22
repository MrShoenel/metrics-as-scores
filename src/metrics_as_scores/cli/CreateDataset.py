"""
This module contains the workflow for creating own datasets.
"""

from typing import Union
from metrics_as_scores.__init__ import DATASETS_DIR, MAS_DIR
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import isint, isnumeric, get_known_datasets
from metrics_as_scores.distribution.distribution import Dataset, LocalDataset
from metrics_as_scores.tools.funcs import natsort, transform_to_MAS_dataset
from questionary import Choice
from re import match
from os import makedirs
from json import dump
from collections import OrderedDict
from shutil import copyfile
from pathlib import Path
import numpy as np
import pandas as pd


default_ds = MAS_DIR.joinpath('./datasets/_default')
"""
:meta private:
"""


class CreateDatasetWorkflow(Workflow):
    __doc__ = """
This workflow creates an own, local dataset from a single data source that can
be read by Pandas from file or URL. The original dataset needs three columns:
For a single sample, one column holds the numeric observation, one holds the
ordinal type (name of feature), and one holds the group associated with it. A
dataset can hold one or more features, but should hold at least two groups in
order to compare distributions of a single sample type across groups.

This workflow creates the entire dataset: The manifest JSON, the parametric fits,
the pre-generated distributions. When done, the dataset is installed, such that it
can be discovered and used by the local Web-Application. If you wish to publish
your dataset so that it can be used by others, start the Bundle-workflow from the
previous menu afterwards.
    """.strip()

    def __init__(self, manifest: LocalDataset=None, org_df: pd.DataFrame=None) -> None:
        super().__init__()
        self.regex_id = r'^[a-z]+[a-z-_\d]*?[a-z\d]+?$'
        self.manifest: LocalDataset = manifest
        self.org_df: pd.DataFrame = org_df
        self.dataset: Dataset = None if manifest is None and org_df is None else Dataset(ds=manifest, df=org_df)
    

    @property
    def dataset_dir(self) -> Path:
        if self.manifest is None:
            raise Exception('A manifest is required.')
        return DATASETS_DIR.joinpath(f'./{self.manifest["id"]}')
    

    @property
    def fits_dir(self) -> Path:
        return self.dataset_dir.joinpath('./fits')
    

    @property
    def densities_dir(self) -> Path:
        return self.dataset_dir.joinpath('./densities')
    

    @property
    def tests_dir(self) -> Path:
        return self.dataset_dir.joinpath('./stat-tests')
    

    @property
    def web_dir(self) -> Path:
        return self.dataset_dir.joinpath('./web')
    

    @property
    def path_manifest(self) -> Path:
        return self.dataset_dir.joinpath('./manifest.json')
    

    @property
    def path_df_data(self) -> Path:
        return self.dataset_dir.joinpath('./org-data.csv')


    @property
    def path_test_ANOVA(self) -> Path:
        return self.tests_dir.joinpath('./anova.csv')
    

    @property
    def path_test_TukeyHSD(self) -> Path:
        return self.tests_dir.joinpath('./tukeyhsd.csv')
    

    @property
    def path_test_ks2samp(self) -> Path:
        return self.tests_dir.joinpath('./ks2samp.csv')

    def _set_ideal_values(self, qtypes: list[str]) -> dict[str, Union[float,int]]:
        type_dict = {}
        for t in qtypes:
            type_dict[t] = None
        
        ready = False
        while not ready:
            use_type_idx = self.ask(
                rtype=int,
                prompt='Select a type to set a default value for:',
                options=list([f'{item[0]}={"None" if item[1] is None else item[1]}' for item in type_dict.items()]))
            ideal = self.q.text(message=f'Set the ideal value for {qtypes[use_type_idx]} (leave blank for None):', validate=lambda s: s == '' or isnumeric(s)).ask()
            if ideal == '':
                ideal = None
            else:
                if isint(ideal):
                    ideal = int(ideal)
                else:
                    ideal = float(ideal)
            type_dict[qtypes[use_type_idx]] = ideal

            ready = not self.q.confirm(message='Continue setting ideal values?', default=True).ask()
        
        return type_dict


    def _transform_dataset(self) -> pd.DataFrame:
        self.q.print('\n' + 10*'-')
        file_type = self.ask(options=[
            'CSV', 'Excel'
        ], prompt='What kind of file do you want to read?', rtype=str)

        path = self.q.text(message='Absolute file path or URL to original file:', validate=lambda s: len(s) > 0).ask().strip()
        df: pd.DataFrame = None
        if file_type == 'CSV':
            df = self._read_csv(path_like=path)
        elif file_type == 'Excel':
            df = self._read_excel(path_like=path)
        
        available_cols = OrderedDict({ k: k for k in df.columns.values.tolist() })
        if len(available_cols) < 2:
            raise Exception(f'Need at least one group and one feature. The loaded data frame only has these columns: [{", ".join(available_cols.keys())}]')
        col_ctx = self.ask(prompt='What is the name of the groups\' column?', options=list(available_cols.keys()), rtype=str)
        del available_cols[col_ctx]

        df_dtypes = df.dtypes.to_dict()
        col_feats = self.q.checkbox(message='Please select all features you would like to include.', choices=[
            Choice(title=f'{feat} [{df_dtypes[feat]}]', value=feat) for feat in available_cols.keys()
        ], validate=lambda str_list: len(str_list) > 0).ask()

        self.print_info(text_normal='Converting data frame ', text_vital='...')

        return transform_to_MAS_dataset(df=df, group_col=col_ctx, feature_cols=col_feats)


    def _read_csv(self, path_like: str) -> pd.DataFrame:
        sep = self.q.text(message='What is the separator used?', default=',', validate=lambda s: s in [',', ';', ' ', '\t']).ask()
        dec = self.q.text(message='What is the decimal point?', default='.', validate=lambda s: s in ['.', ',']).ask()
        return pd.read_csv(filepath_or_buffer=path_like, sep=sep, decimal=dec)


    def _read_excel(self, path_like: str) -> pd.DataFrame:
        sheet = self.q.text(message='Which sheet would you like to read?', default='0').ask()
        if isint(sheet):
            sheet = int(sheet)
        header = self.q.text('Zero-based index of header column?', default='0', validate=isint).ask()
        return pd.read_excel(io=path_like, sheet_name=sheet, header=header)


    def _is_feature_discrete(self, df: pd.DataFrame, col_data: str, col_type: str, use_type: str) -> bool:
        temp = df.loc[df[col_type] == use_type, :]
        vals = temp[col_data].to_numpy()
        # Check if all values are integer.
        return np.all(np.mod(vals, 1) == 0)


    def _create_manifest(self, transformed_df: pd.DataFrame=None) -> tuple[LocalDataset, pd.DataFrame]:
        jsd: LocalDataset = {}

        df: pd.DataFrame = None
        col_data: str = None
        col_type: str = None
        col_ctx: str= None
        if transformed_df is None:
            self.q.print('You are now asked some basic info about the dataset.', style=self.style_mas)
            self.q.print(10*'-')
            file_type = self.ask(options=[
                'CSV', 'Excel'
            ], prompt='What kind of file do you want to read?', rtype=str)
            jsd['origin'] = self.q.text(message='Absolute file path or URL to original file:', validate=lambda s: len(s) > 0).ask().strip()
            if file_type == 'CSV':
                df = self._read_csv(path_like=jsd['origin'])
            elif file_type == 'Excel':
                df = self._read_excel(path_like=jsd['origin'])
            
            available_cols = OrderedDict({ k: k for k in df.columns.values.tolist() })
            col_data = self.ask(prompt='Which column holds the data?', options=list(available_cols.keys()), rtype=str)
            del available_cols[col_data]
            col_type = self.ask(prompt='What is the name of the features\' column?', options=list(available_cols.keys()), rtype=str)
            del available_cols[col_type]
            col_ctx = self.ask(prompt='What is the name of the groups\' column?', options=list(available_cols.keys()), rtype=str)
            del available_cols[col_ctx]
            
            self.q.print('')
            self.print_info(text_normal='Having an original data frame with ', text_vital=f'{len(df.index)} rows.')
        else:
            df = transformed_df
            self.print_info(text_normal='', text_vital=f'Using transformed data frame with {len(df.index)} rows.')
            col_data = 'Value'
            col_type = 'Feature'
            col_ctx = 'Group'
        
        jsd['colname_data'] = col_data
        jsd['colname_type'] = col_type
        jsd['colname_context'] = col_ctx
        

        # Now we only retain those two columns and all complete rows
        df = df.loc[:, [col_data, col_type, col_ctx]]
        df = df.loc[~(df.isna().any(axis=1)), :]
        # Let's convert the feature and group column to string.
        df[col_type] = df[col_type].astype(str)
        df[col_ctx] = df[col_ctx].astype(str)

        
        qtypes = list([str(a) for a in df[col_type].unique()])
        qtypes.sort(key=natsort)
        contexts = list([str(a) for a in df[col_ctx].unique()])
        contexts.sort(key=natsort)
        jsd['contexts'] = contexts
        self.print_info(text_normal='The following features were found: ', text_vital=', '.join(qtypes))
        self.print_info(text_normal='The following groups exist in the data: ', text_vital=', '.join(contexts))

        # Let's ask some descriptions for qtypes and contexts:
        self.q.print('\nYou should now enter a very brief (max. 50 characters) description for each feature.\n')
        jsd['desc_qtypes'] = { qtype: None for qtype in qtypes }
        for qtype in qtypes:
            jsd['desc_qtypes'][qtype] = self.q.text(message=f'Enter a description for {qtype}: ', validate=lambda s: len(s) > 0 and len(s) <= 50).ask().strip()
        
        self.q.print('\nYou can now enter an optional brief (max. 50 characters) description for each group.\n')
        jsd['desc_contexts'] = { ctx: None for ctx in contexts }
        for ctx in contexts:
            temp = self.q.text(message=f'Enter an optional description for {ctx} (empty for none): ', validate=lambda s: len(s) <= 50).ask().strip()
            jsd['desc_contexts'][ctx] = None if len(temp) == 0 else temp

        # Determine if features are discrete or continuous
        jsd['qtypes'] = { t: None for t in qtypes }
        self.q.print('\nChecking data for each type, whether it is discrete or continuous..')
        for t in qtypes:
            # _is_feature_discrete
            self.print_info(text_normal='Checking type: ', text_vital=t, end='')
            is_disc = self._is_feature_discrete(df=df, col_data=col_data, col_type=col_type, use_type=t)
            jsd['qtypes'][t] = 'discrete' if is_disc else 'continuous'
            self.q.print(f' [{jsd["qtypes"][t]}]')

        # Ideal Values:
        self.q.print(f'\nYou can now define custom ideal values for each feature.', style=self.style_mas)
        self.q.print(10*'-')
        self.q.print('''
An ideal (utopian) value for a feature represents the most desirable
value for it. Depending on your use case, it is possible that there is no such
value for each feature. For example, in software metrics, the lowest
possible - and therefore most desirable - value for complexity, is 1. Most other
software metrics, however, do not have such an ideal value. For example, there
is no best value for lines of code (size) of software.
        '''.strip())
        self.q.print(10*'-')
        if self.q.confirm('Define Ideal Values Now?', default=True).ask():
            jsd['ideal_values'] = self._set_ideal_values(qtypes=qtypes)
            self.q.print('\n' + 10*'-' + '\n')
        else:
            jsd['ideal_values'] = { t: None for t in qtypes }

        jsd['name'] = self.q.text(message='Name of the Dataset:', validate=lambda s: len(s) > 0).ask().strip()
        jsd['desc'] = self.q.text(message='Write a short description:', validate=lambda s: len(s) > 0).ask().strip()
        known_datasets = { ds['id']: ds for ds in get_known_datasets() }
        self.print_info(text_normal='Now you need to choose an ID for your dataset. It must NOT be one of these: ', text_vital=', '.join(known_datasets.keys()), arrow='')
        jsd['id'] = self.q.text(message='ID (2+ chars, lowercase, start+end must be letter):', validate=lambda t: match(self.regex_id, t) is not None and not t in known_datasets).ask().strip()
        temp: list[str] = self.q.text(message='Author(s) (first and last, separate authors by comma):', validate=lambda s: len(s) > 0).ask().strip().split(',')
        jsd['author'] = list(a.strip() for a in temp)

        return (jsd, df)


    def _run_statistical_tests(self) -> None:
        self.q.print('We will now perform some statistical tests and summarize the results.')
        
        self.print_info(text_normal='Performing tests: ', text_vital='Analysis of Variance (ANOVA) ...', arrow='\n')
        anova = self.dataset.analyze_ANOVA(qtypes=self.dataset.quantity_types, contexts=list(self.dataset.contexts(include_all_contexts=True)), unique_vals=True)
        file_anova = str(self.path_test_ANOVA)
        anova.to_csv(file_anova, index=False)
        self.print_info(text_normal='Wrote result to: ', text_vital=file_anova)

        self.print_info(text_normal='Performing tests: ', text_vital='Two-Sample Kolmogorov-Smirnov (KS2) ...', arrow='\n')
        ks2samp = self.dataset.analyze_distr(qtypes=self.dataset.quantity_types, use_ks_2samp=True)
        file_ks2samp = str(self.path_test_ks2samp)
        ks2samp.to_csv(file_ks2samp, index=False)
        self.print_info(text_normal='Wrote result to: ', text_vital=file_ks2samp)

        self.print_info(text_normal='Performing tests: ', text_vital="Tukey's Honest Significance Test (TukeyHSD) ...", arrow='\n')
        tukeyhsd = self.dataset.analyze_TukeyHSD(qtypes=self.dataset.quantity_types)
        file_tukey = str(self.path_test_TukeyHSD)
        tukeyhsd.to_csv(file_tukey, index=False)
        self.print_info(text_normal='Wrote result to: ', text_vital=file_tukey)


    def _make_dirs(self) -> None:
        for dir in [self.dataset_dir, self.fits_dir, self.densities_dir, self.tests_dir, self.web_dir]:
            if not dir.exists():
                makedirs(str(dir.resolve()))
    

    def _init_dataset(self) -> None:
        for file in ['./About.qmd', './_quarto.yml', './refs.bib', './web/about.html', './web/references.html']:
            copyfile(src=str(default_ds.joinpath(file)), dst=str(self.dataset_dir.joinpath(file)))
        
        # We also need to write out variables for Quarto.
        # We will write into the _quarto.yaml for better compatibility.
        with open(file=str(self.dataset_dir.joinpath('./_quarto.yml')), mode='a', encoding='utf-8') as fp:
            fp.write(f'\ntitle: {self.manifest["name"]}\n')
            fp.write('\nauthor:')
            for a in self.manifest['author']:
                fp.write(f'\n  - {a}')
    

    def _save_manifest_and_data(self) -> None:
        with open(file=self.path_manifest, mode='w', encoding='utf-8') as fp:
            dump(obj=self.manifest, fp=fp, indent=2)
        self.org_df.to_csv(path_or_buf=self.path_df_data, index=False)


    def create_own(self) -> None:
        """
        Main entry point for this workflow.
        """
        self._print_doc(more='''
You are about to create a new Dataset from a resource that can be read by Pandas
(e.g., a CSV from file or URL). A Dataset that can be used by Metrics-As-Scores
requires a manifest (that will be created as part of this process), parametric
fits, as well as pre-generated distributions that are used in the Web Application.
The following workflow will take you through the creation of the dataset, and other
workflows exist to cover the fitting of random variables and generating densities.''')
        self._wait_for(what_for='to begin')


        self.q.print('')
        self.q.print(text='''
Metrics As Scores requires the data to be in a specific format: A 3-column data frame,
where one column holds the name of the feature, one holding the name of the group, and
one holding the associated value. This special format also allows to mix integral and
real values in the data column, as Metrics As Scores will detect possible integrality
automatically, per feature.

If your data is not in that format, but rather in the much more common format where the
data frame has one column with the group names, and a dedicated column for each feature,
then you have now the choice to transform such a data frame into the format required by
Metrics As Scores before proceeding with the creation process.
'''.strip())

        transformed_df: pd.DataFrame = None
        self.q.print('')
        if self.q.confirm(message='Would you like to transform a data frame first?', default=False).ask():
            transformed_df = self._transform_dataset()

        self.manifest, self.org_df = self._create_manifest(transformed_df=transformed_df)
        self.dataset = Dataset(ds=self.manifest, df=self.org_df)
        
        # Let's create a folder for this dataset (by ID) and out the files there.
        self._make_dirs()
        
        # Let's copy specific files from the default over to the new dataset:
        self._init_dataset()

        # Let's write out the manifest and the dataset:
        self._save_manifest_and_data()

        self.q.print('')
        self.print_info(text_normal='Wrote manifest to: ', text_vital=str(self.path_manifest))
        self.print_info(text_normal='Wrote original dataset to: ', text_vital=str(self.path_df_data))
        self.q.print('\n' + 10*'-' + '\n')

        # Let's run the statistical tests and write them out:
        self._run_statistical_tests()
        self.q.print('\n' + 10*'-' + '\n')


        self.q.print(text=f'''
Your dataset was created and initialized in: {str(self.dataset_dir.resolve())}

In order to use your dataset, you need to pre-generate densities for the Web Application.
If you would also like to fit parametric random variables and generate densities for those,
you need to do that first, however.

If you want to publicize your dataset and share it with others, you need to edit the files
'about.html' and 'references.html' in the /web folder of your newly created dataset. Before
bundling, you must create an 'About.pdf' file. You may use Quarto to knit the default
'About.qmd' that will produce a nice-looking overview and include some qualitative results
of the conducted statistical tests (ANOVA and TukeyHSD).''', style=self.style_mas)
        self.q.print('\n' + 10*'-' + '\n')
