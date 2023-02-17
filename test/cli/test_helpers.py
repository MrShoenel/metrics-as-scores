from metrics_as_scores.cli.helpers import format_file_size, get_known_datasets, get_local_datasets, isint, isnumeric, KNOWN_DATASETS_FILE
from metrics_as_scores.__init__ import DATASETS_DIR
from urllib.request import urlopen




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
    for ds in get_local_datasets():
        assert not ds['name'] == '_default'


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

