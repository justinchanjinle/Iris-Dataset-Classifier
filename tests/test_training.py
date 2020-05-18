import subprocess

from pathlib import Path
from app.src.training import Training

import pytest
import sys

from app.utils.enumerations import Directory, MachineLearningModels


def test_save_model(training: Training, tmp_path: Path):
    save_model_dir = tmp_path / 'random_forest_train.joblib'
    arguments = [sys.executable, '-m', 'scripts.train_model',
                 '--raw_data_dir', str(Directory.IRIS_DATA_DIR.value),
                 '--model', MachineLearningModels.random_forest_default.name,
                 '--model_save_dir', str(save_model_dir)]

    try:
        output = subprocess.check_output(arguments, stderr=subprocess.PIPE)
        print(output.decode("utf-8"))
        assert save_model_dir.exists(), 'Model file directory does not exist'

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Random forest model training failed: \n {e.stderr.decode('utf-8')}")

