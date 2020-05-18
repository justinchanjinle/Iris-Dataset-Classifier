import pandas as pd
import pytest

from app.src.predict import Predict


def test_predict(predict: Predict, x_test: pd.DataFrame):

    try:

        y_predict = predict.predict(x_test)  # noqa

    except Exception as exception:
        pytest.fail(f'Prediction failed: {exception}')
