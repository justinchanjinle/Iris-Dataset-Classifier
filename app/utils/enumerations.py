from enum import Enum
from pathlib import Path


class ModelNames(Enum):

    RANDOM_FOREST_DEFAULT = "random_forest_default.joblib"


class Directory(Enum):

    PARENT_DIR = Path(__file__).parent.parent.resolve()
    MODELS_DIR = PARENT_DIR / "models"
    RANDOM_FOREST_DEFAULT_DIR = MODELS_DIR / ModelNames.RANDOM_FOREST_DEFAULT.value
