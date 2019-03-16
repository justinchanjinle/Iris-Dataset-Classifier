from src.training import Training

import pytest


def test_training(clean_data, random_forest_model):

    try:
        random_forest_model_trained = Training(clean_data, random_forest_model)  # noqa

    except Exception as exception:
        pytest.fail('Random forest model training failed: {}'.format(exception))
