from urllib.error import URLError
from fastapi import APIRouter, HTTPException
from models.schema.census import CensusLink, CensusTrainingResponse
import pandas as pd
import numpy as np
from joblib import load

from models.ml.training import transform_data, evaluate_model

app_census_validate_v1 = APIRouter()


@app_census_validate_v1.post('/census/validate',
                             tags=["Cross Validation"],
                             response_model=CensusTrainingResponse,
                             description="Fetch the data from the link, and cross validate it using available classifier")
async def cross_validation(census: CensusLink):

    try:
        clf = load('models/ml/classifier.pkl')
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail="Model not present, please build a model before prediction")

    path = dict(census)['link']

    try:
        data = pd.read_csv(path, header=None)
    except URLError as e:
        raise HTTPException(
            status_code=400, detail="Bad URL link. Please specify the right link")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400, detail="Bad URL link. Please specify the right link")

    # transform data
    X, y = transform_data(data)

    # model evaluation
    scores = evaluate_model(clf, X, y, cv=5)

    return {"Precision": round(np.mean(scores['test_precision']), 2),
            "Recall": round(np.mean(scores['test_recall']), 2),
            "F1Score": round(np.mean(scores['test_f1_score']), 2)}
