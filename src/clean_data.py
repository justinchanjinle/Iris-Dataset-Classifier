from sklearn.model_selection import train_test_split

from src.ingest_data import IngestData
from utils.enumerations import LabelColumnNames


class CleanData(object):

    def __init__(self, ingest_data: IngestData, test_size=0.2, random_state=0):
        self._ingest_data = ingest_data
        self._model_data = self._ingest_data.get_data_as_dataframe()
        self._test_size = test_size
        self._random_state = random_state

    def get_features_column(self):
        return self._model_data.drop(labels=LabelColumnNames.CLASS.value, axis=1)

    def get_label_column(self):
        return self._model_data[LabelColumnNames.CLASS.value]

    def get_train_and_test_data(self):

        return train_test_split(self.get_features_column(), self.get_label_column(), test_size=self._test_size,
                                random_state=self._random_state)
