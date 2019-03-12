import pandas as pd
import pytest


def test_ingest_data(ingest_data):

    try:

        iris_data = ingest_data.get_data_as_dataframe()

        assert isinstance(iris_data, pd.DataFrame), 'Dataset is not a DataFrame'
        assert not iris_data.empty, 'DataFrame is empty'

    except Exception as exception:
        pytest.fail('Failed ingest data check: {}'.format(exception))
