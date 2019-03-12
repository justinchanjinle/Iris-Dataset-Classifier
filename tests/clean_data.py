import pytest

from utils.enumerations import FEATURE_COLUMN_NAMES_LIST, LabelColumnNames


def test_clean_data(clean_data):

    try:

        y_label = clean_data.get_label_column()
        assert y_label.name == LabelColumnNames.CLASS.value

        x_features = clean_data.get_features_column()
        assert set(list(x_features)) == set(FEATURE_COLUMN_NAMES_LIST), 'Invalid number of features'

    except Exception as exception:
        pytest.fail('Cleaning data failed: {}'.format(exception))
