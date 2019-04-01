import joblib

from pathlib import Path
from typing import TypeVar

from sklearn.base import ClassifierMixin

from src.clean_data import CleanData

TClassifier = TypeVar('TClassifier', bound=ClassifierMixin)


class Training(object):

    def __init__(self, clean_data: CleanData, model: TClassifier, save_model_dir: Path):
        self._clean_data = clean_data
        self._x_train, self._x_test, self._y_train, self._y_test = self._clean_data.get_train_and_test_data()
        self._model = model
        self._save_model_dir = save_model_dir

    @property
    def x_train(self):
        return self._x_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    def train_model(self, **kwargs):
        return self._model.fit(self._x_train, self._y_train, **kwargs)

    def save_model(self, **train_kwargs):

        model_file = self.train_model(**train_kwargs)

        with self._save_model_dir.open('wb') as save_model_file:

            joblib.dump(model_file, save_model_file)
