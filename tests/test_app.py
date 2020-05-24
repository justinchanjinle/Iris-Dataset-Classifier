
def test_home_status(client):

    response = client.get("/")
    assert response.status_code == 200


def test_prediction(client):
    data = {"sepal_length": [4.9],
            "sepal_width": [3.0],
            "petal_length": [1.4],
            "petal_width": [0.2]}
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
