from src.ingest_data import IngestData
from utils.enumerations import LabelColumnNames


class CleanData(object):

    def __init__(self, ingest_data: IngestData):
        self._ingest_data = ingest_data
        self._model_data = self._ingest_data.get_data_as_dataframe()

    def get_features_column(self):
        return self._model_data.drop(labels=LabelColumnNames.CLASS.value, axis=1)

    def get_label_column(self):
        return self._model_data[LabelColumnNames.CLASS.value]
