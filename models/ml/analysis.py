import pandas as pd
from models.ml.training import transform_data, build_model, evaluate_model
import numpy as np

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz'
data = pd.read_csv(path, header=None)

# transform data
X, y = transform_data(data)

# model building
pipe = build_model(X, y)

# model evaluation
scores = evaluate_model(pipe, X, y, cv=5)

scores_dict = {"Precision": round(np.mean(scores['test_precision']), 2),
               "Recall": round(np.mean(scores['test_recall']), 2),
               "F1Score": round(np.mean(scores['test_f1_score']), 2),
               "Accuracy": round(np.mean(scores['test_accuracy']), 2)}

print(scores_dict)

