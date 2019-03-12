from src.clean_data import CleanData


class Training(object):

    def __init__(self, clean_data: CleanData, model):
        self._clean_data = clean_data
        self._x_train, self._x_test, self._y_train, self._y_test = self._clean_data.get_train_and_test_data()
        self._model = model

    def train_model(self, **kwargs):
        return self._model.fit(self._x_train, self._y_train, **kwargs)

