# Census Income API

## Dependencies

- Python
- Git
- Docker
- Heroku CLI (deployment)

## Steps

1. Clone the repository

    `git clone https://github.com/vivekmehendiratta/census_income_api.git`

2. Create and activate virtual environment

    `pip install virtualenv`

    `python -m venv myenv`

    `.\myenv\Scripts\activate`

3. Install requirements

    `pip install -r requirements.txt`

4. Run project

    `python app.py`

5. API testing

    - Go to `localhost:80/docs`

    - Request body for Training - `/v1/census/train` :

        - `{"link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"}`

    - Request body for Validation - `/v1/census/validate` :

        - `{"link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"}`

    - Request body for Predictions - `/v1/census/predict` :

        - Single data point

6. Load Testing

    `locust -f ./tests/load_test.py`

7. Deployment

    - Create and Run docker images

        `docker build -t <image-name> .`

        `docker run -d --name my_container -p 80:80 <image-name>`

    - Deployment using Heroku or any cloud service. Unformatunaltely, Heroku is down and ACR requires premium subscription. Left it for future scope.
