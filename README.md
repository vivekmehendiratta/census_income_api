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

    `uvicorn app:app --host 0.0.0.0 --port 80`

5. API testing

    - Go to `localhost:80/docs`

    - Request body for `/v1/census/train`

        - `{"link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"}`

    - Request body for `/v1/census/predict`

        - `{"link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz"}`

6. Deployment

    - Create and Run docker images

        `docker build -t <image-name> .`

        `docker run -d --name my_container -p 80:80 <image-name>`

    - Deployment using Heroku or any cloud service. Unformatunaltely, Heroku is down and ACR requires premium subscription. Left it for future scope.