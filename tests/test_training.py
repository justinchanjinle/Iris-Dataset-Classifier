import subprocess

from pathlib import Path
from app.src.training import Training

import pytest

from app.utils.enumerations import Directory, MachineLearningModels


def test_save_model(training: Training, tmp_path: Path):

    try:

        save_model_dir = tmp_path / 'random_forest_train.joblib'

        subprocess.run(['python3', '-m', 'scripts.train_model',
                        '--raw_data_dir', Directory.IRIS_DATA_DIR.value,
                        '--model', MachineLearningModels.random_forest_default.name,
                        '--model_save_dir', save_model_dir], cwd=Directory.PARENT_DIR.value)

        assert save_model_dir.exists(), 'Model file directory does not exist'

    except Exception as exception:
        pytest.fail('Random forest model training failed: {}'.format(exception))
