from typing import Iterable, Literal, Union
from typing import TypedDict


class JsonDataset(TypedDict):
    name: str
    desc: str
    id: str
    author: list[str]
    ideal_values: dict[str, Union[int, float]]


class LocalDataset(JsonDataset):
    origin: str
    colname_data: str
    colname_type: str
    colname_context: str
    qtypes: dict[str, Literal['continuous', 'discrete']]


class KnownDataset(JsonDataset):
    info_url: str
    download: str



def isint(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False

def isnumeric(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False


class Dataset:
    def __init__(self, name: str, id: str) -> None:
        self.name = name
        self.id = id
        pass


    @staticmethod
    def get_available_datasets() -> Iterable['Dataset']:
        yield Dataset('Qualitas.class Corpus', 'qcc')
        yield Dataset('ELISA HIV Samples', 'elisa')
