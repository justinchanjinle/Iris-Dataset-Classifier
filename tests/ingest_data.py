import pandas as pd
import pytest


def test_ingest_data(ingest_data):

    try:

        iris_data = ingest_data.get_data_as_dataframe()

        assert isinstance(iris_data, pd.DataFrame), 'Dataset is not a DataFrame'

    except Exception as exception:
        pytest.fail('Failed to ingest data: {}'.format(exception))
