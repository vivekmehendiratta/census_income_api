import pandas as pd
from models.ml.training import transform_data, build_model, evaluate_model
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, average_precision_score
import numpy as np
from matplotlib import pyplot as plt
import json


path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz'
data = pd.read_csv(path, header=None)

# transform data
X, y = transform_data(data)

# model building
pipe = build_model(X, y)

prediction = pipe.predict(X).tolist()
prediction_prob = pipe.predict_proba(X).tolist()
# prediction = [0]*len(X)

f1 = f1_score(y, prediction, average='weighted')
precision = precision_score(y, prediction, average='weighted')
recall = recall_score(y, prediction, average='weighted')
acc = accuracy_score(y, prediction)

# confusion matrix
ConfusionMatrixDisplay.from_estimator(
    pipe, X, y, cmap="YlGn", normalize="true"
)
plt.savefig('CM.png')

# PR curve
display = PrecisionRecallDisplay.from_estimator(
    pipe, X, y, name="XGBoost"
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.savefig("PR_curve.png")

# Metrics
scores_dict = {"Precision": round(precision, 4),
               "Recall": round(recall, 4),
               "F1Score": round(f1, 4),
               "Accuracy"  :round(acc, 4),
               "AVG Precision Score" : round(display.average_precision, 4)}

# print(scores_dict)

## metrics report
with open("metrics.json", 'w') as out: 
    json.dump(scores_dict, out)
    # for key, value in scores_dict.items(): 
    #     f.write('%s : %s\n' % (key, value))
