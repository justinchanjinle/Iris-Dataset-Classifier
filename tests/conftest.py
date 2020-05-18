from enum import Enum
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.src.clean_data import CleanData
from ml_pipeline.src.ingest_data import IngestData
from app.src.predict import Predict
from ml_pipeline.src.training import Training
from ml_pipeline.utils.enumerations import Directory, FolderNames


class Directories(Enum):

    PARENT_DIR = Path(__file__).parent.resolve()

    MODELS_DIR = PARENT_DIR / FolderNames.MODELS.value

    RF_MODEL_PREDICT_DIR = MODELS_DIR / 'random_forest_predict.joblib'


@pytest.fixture(scope='module')
def ingest_data():
    return IngestData(Directory.IRIS_DATA_DIR.value)


@pytest.fixture(scope='module')
def clean_data(ingest_data: IngestData):
    return CleanData(ingest_data)


@pytest.fixture(scope='module')
def random_forest_model():
    return RandomForestClassifier(max_depth=20, random_state=0, n_estimators=200, n_jobs=6)


@pytest.fixture(scope='module')
def training(clean_data: CleanData, random_forest_model):
    return Training(clean_data, random_forest_model)


@pytest.fixture(scope='module')
def x_test(training: Training):
    return training.x_test


@pytest.fixture(scope='module')
def predict():
    return Predict(Directories.RF_MODEL_PREDICT_DIR.value)
