from pathlib import Path

import joblib


class Predict(object):

    def __init__(self, load_model_dir: Path):
        self._load_model_dir = load_model_dir

    def predict(self, x_data):

        trained_model = self._load_model()
        return trained_model.predict(x_data)

    def _load_model(self):

        with self._load_model_dir.open('rb') as model_file:
            return joblib.load(model_file)
