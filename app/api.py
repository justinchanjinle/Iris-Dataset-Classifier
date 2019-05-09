from flask import Flask, jsonify, make_response, request, Response
from pandas import DataFrame

from src.predict import Predict
from utils.enumerations import Directory

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict() -> Response:
    """Set up a rest API to predicts a given set of features"""

    try:
        random_forest = Predict(Directory.RANDOM_FOREST_DEFAULT_DIR.value)
        features = DataFrame(request.json)
        prediction = random_forest.predict(features).tolist()
        return make_response(jsonify({'prediction': prediction}))

    except Exception as exception:
        raise RuntimeError(f'Failed to predict iris classes: {exception}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
