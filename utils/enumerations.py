from enum import Enum
from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent.resolve()


class FolderNames(Enum):

    DATA = 'data'


class FileNames(Enum):

    IRIS_DATA = 'iris.csv'


class Directory(Enum):

    IRIS_DATA_DIR = PARENT_DIR / FolderNames.DATA.value / FileNames.IRIS_DATA.value
