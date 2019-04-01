import subprocess

from src.training import Training

import pytest

from tests.conftest import DirectoryTest
from utils.enumerations import Directory, Models


def test_save_model(training: Training):

    try:

        save_model_dir = DirectoryTest.RF_MODEL_TRAIN_DIR.value

        if save_model_dir.exists():
            save_model_dir.unlink()

        subprocess.run(['python3.6', '-m', 'scripts.train_model',
                        '--raw_data_dir', Directory.IRIS_DATA_DIR.value,
                        '--model', Models.random_forest_default.name,
                        '--model_save_dir', save_model_dir], cwd=Directory.PARENT_DIR.value)

        assert save_model_dir.exists(), 'Model file directory does not exist'

    except Exception as exception:
        pytest.fail('Random forest model training failed: {}'.format(exception))
