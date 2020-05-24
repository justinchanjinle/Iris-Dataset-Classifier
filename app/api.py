from flask import Flask, jsonify, make_response, request
from flask_restx import Api, Resource, fields
from pandas import DataFrame

from app.src.predict import Predict
from app.utils.enumerations import Directory

app = Flask(__name__)
api = Api(app=app, version="1.0", title="Iris Classifier", description="Predict iris type based on data")

name_space = api.namespace("predict", description="Predict Iris type.")

model = api.model("Prediction params",
                  {"sepal_length": fields.List(fields.Float, required=True,
                                               description="Sepal Length"),
                   "sepal_width": fields.List(fields.Float, required=True,
                                              description="Sepal Width"),
                   "petal_length": fields.List(fields.Float, required=True,
                                               description="Petal Length"),
                   "petal_width": fields.List(fields.Float, required=True,
                                              description="Petal Width")})


@name_space.route("/")
class PredictIris(Resource):

    @staticmethod
    @name_space.doc(responses={200: "Prediction successful",
                               400: "Invalid prediction parameters",
                               500: "Prediction failed"})
    @name_space.expect(model)
    def post():
        """Set up a rest API to predicts a given set of features"""

        try:
            random_forest = Predict(Directory.RANDOM_FOREST_DEFAULT_DIR.value)
            features = DataFrame(request.json)
            prediction = random_forest.predict(features).tolist()
            return make_response(jsonify({"prediction": prediction}))

        except Exception as exception:
            raise RuntimeError(f"Failed to predict iris classes: {exception}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
