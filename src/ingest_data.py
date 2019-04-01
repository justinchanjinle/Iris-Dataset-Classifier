from pathlib import Path

import pandas as pd

from utils.enumerations import FeatureColumnNames, LabelColumnNames


class IngestData(object):

    def __init__(self, data_directory: Path):
        self._data_directory = data_directory

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._data_directory, index_col=False, names=[FeatureColumnNames.SEPAL_LENGTH.value,
                                                                         FeatureColumnNames.SEPAL_WIDTH.value,
                                                                         FeatureColumnNames.PETAL_LENGTH.value,
                                                                         FeatureColumnNames.PETAL_WIDTH.value,
                                                                         LabelColumnNames.CLASS.value])
