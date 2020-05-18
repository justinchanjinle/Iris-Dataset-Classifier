import pytest

from ml_pipeline.utils.enumerations import FEATURE_COLUMN_NAMES_LIST, LabelColumnNames


def test_clean_data(clean_data):

    try:

        # Check y_label is returned correctly
        y_label = clean_data.get_label_column()
        assert y_label.name == LabelColumnNames.CLASS.value

        # Check x_features is returned correctly
        x_features = clean_data.get_features_column()
        assert set(list(x_features)) == set(FEATURE_COLUMN_NAMES_LIST), 'Invalid number of features'

        x_train, x_test, y_train, y_test = clean_data.get_train_and_test_data()  # noqa

    except Exception as exception:
        pytest.fail('Cleaning data failed: {}'.format(exception))
