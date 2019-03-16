import pytest
from sklearn.ensemble import ExtraTreesClassifier

from src.clean_data import CleanData
from src.ingest_data import IngestData
from src.training import Training
from utils.enumerations import Directory


@pytest.fixture(scope='module')
def ingest_data():
    return IngestData(Directory.IRIS_DATA_DIR)


@pytest.fixture(scope='module')
def clean_data(ingest_data: IngestData):
    return CleanData(ingest_data)


@pytest.fixture(scope='module')
def random_forest_model():
    return ExtraTreesClassifier(max_depth=20, random_state=0, n_estimators=200, n_jobs=6)


@pytest.fixture(scope='module')
def training(clean_data: CleanData, random_forest_model):
    return Training(clean_data, random_forest_model)
