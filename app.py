from fastapi import FastAPI
from joblib import load
from routes.v1.census_predict import app_census_predict_v1
from routes.v1.census_train import app_census_train_v1
from routes.home import app_home

app = FastAPI(title="Census Income ML API", description="API for census income dataset ml model", version="1.0")


# @app.on_event('startup')
# async def load_model():
    
#     try:
#         clf = load('models/ml/classifier.pkl')
#     except FileNotFoundError as e:
#         print("No model present as of now, please build a model")


app.include_router(app_home)
app.include_router(app_census_predict_v1, prefix='/v1')
app.include_router(app_census_train_v1, prefix='/v1')