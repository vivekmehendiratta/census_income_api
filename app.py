from fastapi import FastAPI
from routes.v1.census_predict import app_census_predict_v1
from routes.v1.census_train import app_census_train_v1
from routes.v1.census_validate import app_census_validate_v1
from routes.home import app_home

import uvicorn

app = FastAPI(title="Census Income ML API", description="API for census income dataset ml model", version="1.0")

app.include_router(app_home)
app.include_router(app_census_predict_v1, prefix='/v1')
app.include_router(app_census_train_v1, prefix='/v1')
app.include_router(app_census_validate_v1, prefix = '/v1')

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0')