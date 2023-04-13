from metrics_as_scores.cli.helpers import format_file_size, get_known_datasets, get_local_datasets, isint, isnumeric, required_files_folders_local_dataset, validate_local_dataset_files, PathStatus, KNOWN_DATASETS_FILE
from metrics_as_scores.__init__ import DATASETS_DIR, MAS_DIR
from urllib.request import urlopen
from shutil import copyfile, copytree, rmtree
from uuid import uuid4
from pathlib import Path
from os import makedirs




def test_isint():
    for i in ['0', '1', '2341345', '-324', '-0']:
        assert isint(i)
    
    for ni in ['asdf', '0.5', True, None, '1e3', '-1.2e-3', object()]:
        assert not isint(ni)


def test_isnumeric():
    for n in ['0', '1', '2341345', '-324', '-0', '0.5', '1e3', '-1.2e-3']:
        assert isnumeric(n)
    
    for ni in ['asdf', True, None, object()]:
        assert not isnumeric(ni)


def test_local_datasets():
    src_default = MAS_DIR.joinpath('./datasets/_default')
    dst_default = DATASETS_DIR.joinpath('./_default')

    if not src_default.exists() or not src_default.is_dir():
        raise Exception('The _default directory does not exist')
    
    default_existed = True
    if str(src_default.resolve()) != str(dst_default.resolve()):
        # Conditionally copy over _default
        if not dst_default.exists():
            default_existed = False
            copytree(src=str(src_default), dst=str(dst_default), dirs_exist_ok=False)
    
    # Also make a temporary dataset:
    temp_id = str(uuid4())
    dst_temp = DATASETS_DIR.joinpath(f'./{temp_id}')
    copytree(src=str(src_default), dst=str(dst_temp), dirs_exist_ok=False)
    # Create a dir as a file and a file as a dir:
    makedirs(name=str(dst_temp.joinpath('./org-data.csv')), exist_ok=False)
    dst_temp.joinpath('./fits').touch(exist_ok=False)
    # There is no manifest, so the validator should find that:
    dirs, files = validate_local_dataset_files(*required_files_folders_local_dataset(local_ds_id=temp_id))
    for d in dirs:
        if 'web' in str(d): # Only 'web' should exist at this point.
            assert dirs[d] == PathStatus.OK
        elif 'fits' in str(d):
            assert dirs[d] == PathStatus.NOT_A_DIRECTORY
        else:
            assert dirs[d] == PathStatus.DOESNT_EXIST
    for f in files:
        if 'refs.bib' in str(f) or 'about.html' in str(f) or 'references.html' in str(f):
            assert files[f] == PathStatus.OK
        elif 'org-data.csv' in str(f):
            assert files[f] == PathStatus.NOT_A_FILE
        else:
            assert files[f] == PathStatus.DOESNT_EXIST


    # There is no manifest in this one, let's copy one:
    this_dir = Path(__file__).parent
    qcc_manifest_file = this_dir.parent.joinpath('./data/qcc-manifest.json')
    copyfile(src=str(qcc_manifest_file.resolve()), dst=str(dst_temp.joinpath('./manifest.json').resolve()))

    ids = []
    for ds in get_local_datasets():
        ids.append(ds['id'])
    assert 'qcc' in set(ids)
    assert not '_default' in set(ids)

    rmtree(path=str(dst_temp.resolve()))
    if str(src_default.resolve()) != str(dst_default.resolve()) and not default_existed:
        rmtree(path=str(dst_default.resolve()))


def test_known_datasets():
    known_file_local = DATASETS_DIR.joinpath('./known-datasets.json')
    if not known_file_local.exists():
        with open(file=str(known_file_local), mode='w', encoding='utf-8') as fp:
            fp.write(urlopen(url=KNOWN_DATASETS_FILE).read().decode('utf-8'))

    for ds in get_known_datasets(use_local_file=True):
        assert isinstance(ds['name'], str)
    for ds in get_known_datasets(use_local_file=False):
        assert isinstance(ds['name'], str)


def test_format_file_size():
    ffs = format_file_size
    assert ffs(12) == '12 B'
    assert ffs(1_234_567) == '1.23 MB'
    assert ffs(1_234_567, digits=0) == '1 MB'
    assert ffs(1_234_567_899, digits=1) == '1.2 GB'

