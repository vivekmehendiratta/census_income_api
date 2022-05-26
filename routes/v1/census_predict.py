from fastapi import APIRouter, HTTPException
from models.schema.census import CensusDataPointList, CensusJsonPredictionResponse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from joblib import load
from models.ml.training import transform_data 
import pandas as pd

app_census_predict_v1 = APIRouter()


@app_census_predict_v1.post('/census/predict',
                          tags=["Predictions"],
                          response_model=CensusJsonPredictionResponse,
                          description="Get a predictions for given data points using existing classifier")
async def get_prediction(dataArray: CensusDataPointList):

    try:
        clf = load('models/ml/classifier.pkl')
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="Model not present, please build a model before prediction")

    data = pd.DataFrame(dataArray.dict()['dataArray'])

    # transform data
    X, _ = transform_data(data, request_type='predict')

    prediction = clf.predict(X).tolist()

    return {"prediction" : prediction}