from locust import HttpUser, TaskSet, task, between, tag

"""
Run locus with:
locust -f ./tests/load_test.py
"""

class CensusTrain(TaskSet):
    @tag('Predictions')
    @task
    def predict(self):
        request_body = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"
        self.client.post('/v1/census/train', json=request_body)

    @tag('Baseline')
    @task
    def health_check(self):
        self.client.get('/')

class CensusPredict(TaskSet):
    @tag('Predictions')
    @task
    def predict(self):
        request_body = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz"
        self.client.post('/v1/census/predict', json=request_body)

    @tag('Baseline')
    @task
    def health_check(self):
        self.client.get('/')


class CensusLoadTest(HttpUser):
    tasks = [CensusPredict]
    host = 'http://127.0.0.1'
    stop_timeout = 200
    wait_time = between(1, 5)