from fastapi.testclient import TestClient
from app import app


def test_success_prediction():
    endpoint = '/v1/census/predict'
    body = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz"

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        response_json = response.json()
        assert response.status_code == 200
        assert 'prediction' in response_json


def test_bad_request():
    endpoint = '/v1/census/predict'
    body = "asjkbas"

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 400