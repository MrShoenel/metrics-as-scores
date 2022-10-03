from typing import Callable
from poetry.utils._compat import metadata

version: Callable[[str], str] = metadata.version

__version__ = version("metrics-as-scores")