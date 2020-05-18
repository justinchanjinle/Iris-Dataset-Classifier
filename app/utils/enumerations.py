from enum import Enum
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


class FolderNames(Enum):

    DATA = 'data'
    MODELS = 'models'


class FileNames(Enum):

    IRIS_DATA = 'iris.csv'


class ModelNames(Enum):

    RANDOM_FOREST_DEFAULT = 'random_forest_default.joblib'


class Directory(Enum):

    PARENT_DIR = Path(__file__).parent.parent.parent.resolve()
    APP_PARENT_DIR = PARENT_DIR / "app"
    IRIS_DATA_DIR = APP_PARENT_DIR / FolderNames.DATA.value / FileNames.IRIS_DATA.value
    RANDOM_FOREST_DEFAULT_DIR = APP_PARENT_DIR / FolderNames.MODELS.value / ModelNames.RANDOM_FOREST_DEFAULT.value


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


class MachineLearningModels(Enum):

    random_forest_default = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=200, n_jobs=6)


MODELS_NAME_LIST = [model.name for model in MachineLearningModels]
