from src.training import Training

import pytest

from tests.conftest import DirectoryTest


def test_save_model(training: Training):

    try:

        save_model_dir = DirectoryTest.RF_MODEL_TRAIN_DIR.value

        if save_model_dir.exists():
            save_model_dir.unlink()

        training.save_model()

        assert save_model_dir.exists(), 'Model file directory does not exist'

    except Exception as exception:
        pytest.fail('Random forest model training failed: {}'.format(exception))
