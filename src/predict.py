from src.training import Training


class Predict(object):

    def __init__(self, training: Training):
        self._training = training
        self._model = self._training.train_model()

    def predict(self, x_data):
        return self._model.predict(x_data)
