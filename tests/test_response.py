from fastapi.testclient import TestClient
from app import app

def train_success():
    endpoint = '/v1/census/train'
    body = { 
        "link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"
    }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        response_json = response.json()
        assert response.status_code == 200


def test_bad_request():
    endpoint = '/v1/census/train'
    body = { "link": "asjkbas" }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 400

def predict_success():
    endpoint = '/v1/census/predict'
    body = { 
        "link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz"
    }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 200


def test_predict_bad_request():
    endpoint = '/v1/census/predict'
    body = { "link": "asjkbas" }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 400