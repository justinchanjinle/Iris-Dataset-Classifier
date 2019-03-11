import pandas as pd

from utils.enumerations import Directory


class IngestData(object):

    def __init__(self, data_directory: Directory):
        self.data_directory = data_directory

    def get_data_as_dataframe(self):
        return pd.read_csv(self.data_directory.value, index_col=False)
