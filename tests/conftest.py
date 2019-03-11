import pytest

from src.ingest_data import IngestData
from utils.enumerations import Directory


@pytest.fixture(scope='module')
def ingest_data():
    return IngestData(Directory.IRIS_DATA_DIR)
