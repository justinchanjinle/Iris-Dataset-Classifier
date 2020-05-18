import joblib

from pathlib import Path
from typing import TypeVar

from sklearn.base import ClassifierMixin

from app.src.clean_data import CleanData

TClassifier = TypeVar('TClassifier', bound=ClassifierMixin)


class Training(object):

    def __init__(self, clean_data: CleanData, model: TClassifier):
        self._clean_data = clean_data
        self._x_train, self._x_test, self._y_train, self._y_test = self._clean_data.get_train_and_test_data()
        self._model = model

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

    def _train_model(self, **kwargs):
        return self._model.fit(self._x_train, self._y_train, **kwargs)

    def save_model(self,  save_model_dir: Path, **train_kwargs):

        model_file = self._train_model(**train_kwargs)
        with save_model_dir.open('wb') as save_model_file:
            joblib.dump(model_file, save_model_file)
