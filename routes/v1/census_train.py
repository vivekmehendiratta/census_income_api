from urllib.error import URLError
from fastapi import APIRouter, HTTPException
from models.schema.census import CensusLink, CensusTrainingResponse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.ml.training import build_model, transform_data, evaluate_model, save_model

app_census_train_v1 = APIRouter()


@app_census_train_v1.post('/census/train',
                          tags=["Training"],
                          response_model=CensusTrainingResponse,
                          description="Get a data and train the model")
async def model_training(census: CensusLink):

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

    # model building
    pipe = build_model(X, y)

    prediction = pipe.predict(X).tolist()

    f1 = f1_score(y, prediction, average='weighted')
    precision = precision_score(y, prediction, average='weighted')
    recall = recall_score(y, prediction, average='weighted')

    # save model
    save_model(pipe, 'models/ml/classifier.pkl')

    return {
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1Score': round(f1, 2)
    }
