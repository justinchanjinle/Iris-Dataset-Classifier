from enum import Enum
from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent.resolve()


class FolderNames(Enum):

    DATA = 'data'


class FileNames(Enum):

    IRIS_DATA = 'iris.csv'


class Directory(Enum):

    IRIS_DATA_DIR = PARENT_DIR / FolderNames.DATA.value / FileNames.IRIS_DATA.value


class DataColumnNames(Enum):
    pass


class LabelColumnNames(DataColumnNames):
    CLASS = 'class'


class FeatureColumnNames(DataColumnNames):
    SEPAL_LENGTH = 'sepal_length'
    SEPAL_WIDTH = 'sepal_width'
    PETAL_LENGTH = 'petal_length'
    PETAL_WIDTH = 'petal_width'


FEATURE_COLUMN_NAMES_LIST = [column_name.value for column_name in list(FeatureColumnNames)]
