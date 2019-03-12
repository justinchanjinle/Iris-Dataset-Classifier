import pytest

from src.clean_data import CleanData
from src.ingest_data import IngestData
from utils.enumerations import Directory


@pytest.fixture(scope='module')
def ingest_data():
    return IngestData(Directory.IRIS_DATA_DIR)


@pytest.fixture(scope='module')
def clean_data(ingest_data: IngestData):
    return CleanData(ingest_data)
