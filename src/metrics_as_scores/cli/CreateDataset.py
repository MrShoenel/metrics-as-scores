from typing import Literal, Union
from Workflow import Workflow
from helpers import LocalDataset, isint, isnumeric
from re import match
from os import makedirs
from json import dump
import numpy as np
import pandas as pd

from pathlib import Path
this_dir = Path(__file__).resolve().parent
datasets_dir = this_dir.parent.parent.parent.joinpath('./datasets')


class CreateDatasetWorkflow(Workflow):
    """
    This workflow creates an own, local dataset from a single data source that can
    be read by Pandas from file or URL. The original dataset needs three columns:
    For a single sample, one column holds the numeric observation, one holds the
    ordinal type, and one holds the context associated with it. A dataset can hold
    one or more types, but should hold at least two contexts in order to compare
    distributions of a single sample type across contexts.
    This workflow creates the entire dataset: The manifest JSON, the parametric fits,
    the pre-generated distributions. When done, the dataset is installed, such that it
    can be discovered and used by the local Web-Application. If you wish to publish
    your dataset so that it can be used by others, start the Bundle-workflow from the
    previous menu afterwards.
    """
    def __init__(self) -> None:
        super().__init__()
        self.regex_id = r'^[a-z]+[a-z-_\d]*?[a-z\d]+?$'
    

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

    

    def _create_manifest(self) -> tuple[LocalDataset, pd.DataFrame]:
        jsd: LocalDataset = {}

        df: pd.DataFrame = None
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
        
        available_cols = df.columns.values.tolist()
        col_data = self.ask(prompt='Which column holds the data?', options=available_cols, rtype=str)
        jsd['colname_data'] = col_data
        col_type = self.ask(prompt='What is the name of the type column?', options=list(set(available_cols) - set([col_data])), rtype=str)
        jsd['colname_type'] = col_type
        col_ctx = self.ask(prompt='What is the name of the context column?', options=list(set(available_cols) - set([col_data, col_type])), rtype=str)
        jsd['colname_context'] = col_ctx

        # Now we only retain those two columns and all complete rows
        df = df.loc[:, [col_data, col_type, col_ctx]]
        df = df.loc[~(df.isna().any(axis=1)), :]
        # Let's convert the type and context column to string.
        df[col_type] = df[col_type].astype(str)
        df[col_ctx] = df[col_ctx].astype(str)

        self.q.print('')
        self.print_info(text_normal='Having an original data frame with ', text_vital=f'{len(df.index)} rows.')
        qtypes = list([str(a) for a in df[col_type].unique()])
        contexts = list([str(a) for a in df[col_ctx].unique()])
        self.print_info(text_normal='The following quantity types were found: ', text_vital=', '.join(qtypes))
        self.print_info(text_normal='The following contexts exist in the data: ', text_vital=', '.join(contexts))

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
        self.q.print(f'\nYou can now define custom ideal values for each type of quantity.', style=self.style_mas)
        self.q.print(10*'-')
        self.q.print('''
An ideal (utopian) value for a type of quantity represents the most desirable
value for it. Depending on your use case, it is possible that there is no such
value for each type of quantity. For example, in software metrics, the lowest
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
        jsd['id'] = self.q.text(message='ID (2+ chars, lowercase, start+end must be letter):', validate=lambda t: match(self.regex_id, t) is not None).ask().strip()
        temp: list[str] = self.q.text(message='Author(s) (first and last, separate authors by comma):', validate=lambda s: len(s) > 0).ask().strip().split(',')
        jsd['author'] = list(a.strip() for a in temp)

        return (jsd, df)
    

    def create_own(self) -> None:
        self.q.print('\n' + 10*'-')
        self.q.print('''
You are about to create a new Dataset from a resource that can be
read by Pandas (e.g., a CSV from file or URL). A Dataset that can
be used by Metrics-As-Scores requires a manifest (that will be
created as part of this process), optional parametric fits, as well
as pre-generated distributions that are used in the Web-Application.
The following workflow will take you through all of these steps. It
cannot be resumed.
'''.strip())
        self.q.print(10*'-' + '\n')

        manifest, df = self._create_manifest()
        # Let's create a folder for this dataset (by ID) and out the files there.
        dataset_dir = datasets_dir.joinpath(f'./{manifest["id"]}')
        if not dataset_dir.exists():
            makedirs(str(dataset_dir.resolve()))
        
        path_manifest = str(dataset_dir.joinpath('./manifest.json'))
        path_data = str(dataset_dir.joinpath('./org-data.csv'))
        
        with open(file=path_manifest, mode='w', encoding='utf-8') as fp:
            dump(obj=manifest, fp=fp, indent=2)
        df.to_csv(path_or_buf=path_data)

        self.q.print('')
        self.print_info(text_normal='Wrote manifest to: ', text_vital=path_manifest)
        self.print_info(text_normal='Wrote original dataset to: ', text_vital=path_data)
        self.q.print('\n' + 10*'-' + '\n')
    
