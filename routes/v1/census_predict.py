from fastapi import APIRouter, HTTPException
from models.schema.census import CensusLink, CensusPredictionResponse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from joblib import load
from models.ml.training import transform_data 
import pandas as pd
from urllib.error import URLError

app_census_predict_v1 = APIRouter()


@app_census_predict_v1.post('/census/predict',
                          tags=["Predictions"],
                          response_model=CensusPredictionResponse,
                          description="Get a predictions from classifier")
async def get_prediction(census: CensusLink):
    try:
        clf = load('models/ml/classifier.pkl')
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="Model not present, please build a model before prediction")

    path = dict(census)['link']

    try:
        data = pd.read_csv(path, header = None)
    except URLError as e:
        raise HTTPException(status_code=400, detail="Bad URL link. Please specify the right link")

    # transform data
    X, y = transform_data(data)

    prediction = clf.predict(X).tolist()

    f1        = f1_score(y,prediction, average='weighted')
    precision = precision_score(y, prediction, average='weighted')
    recall    = recall_score(y, prediction, average='weighted')
    accuracy  = accuracy_score(y,prediction)

    return {"Precision": round(precision,2),
            "Recall": round(recall,2),
            "F1Score": round(f1,2),
            "Accuracy": round(accuracy,2)}