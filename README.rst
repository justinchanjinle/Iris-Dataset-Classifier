.. image:: https://circleci.com/gh/justinchanjinle/Iris-Dataset-Classifier.svg?style=svg
    :target: https://circleci.com/gh/justinchanjinle/Iris-Dataset-Classifier

Iris Dataset Classifier
^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :depth: 3

Introduction
============

This repository deploys the Iris Dataset Classifier model using docker. The model can be used to predict the type of
iris given the sepal length, sepal width, petal length and petal width.

Usage
=====

1. Pull the docker image from the repository with the following command:

::

    docker pull justinchanjinle/iris-dataset-classifier

2. Run the application in the container

::

    docker run --name iris-predictor -p 5000:5000 -d justinchanjinle/iris-dataset-classifier

Run this command as the first step to run the container immediately

3. Predict the iris type by sending prediction data through the rest api:

::

    curl http://localhost:5000/predict --request POST --header "Content-Type: application/json" --data '{"sepal_length": [4.9], "sepal_width": [3.0], "petal_length": [1.4], "petal_width": [0.2]}'

4. Use of Swagger UI is also available for execution at http://localhost:5000

Dataset
=======
The dataset is obtained from the following site:
https://archive.ics.uci.edu/ml/datasets/iris
