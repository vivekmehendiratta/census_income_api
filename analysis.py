import pandas as pd
from models.ml.training import transform_data, build_model, evaluate_model
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
import numpy as np
from matplotlib import pyplot as plt


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
               "F1Score": round(np.mean(scores['test_f1_score']), 2)}

print(scores_dict)

## metrics report
with open("metrics.txt", 'w') as f: 
    for key, value in scores_dict.items(): 
        f.write('%s:%s\n' % (key, value))


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
