import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, RepeatedKFold


# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = ["BMI", "MCP.1", "Age", "Adiponectin", "Leptin", "Classification"], axis=1)  # axis = 1 (specify column)
y = df['Classification']

# use Kfold

print(x.describe())
# scoring
scoring = ["accuracy", "f1", "precision"]
# use svm

classifier = make_pipeline(StandardScaler(), SVC(kernel="sigmoid", C=0.6,gamma="auto"))

# use cross validation

scores = []

for i in range(3,10):
    for j in range(1, 10):
        kf = RepeatedKFold(n_splits=i, n_repeats=j, random_state=1)
        score=cross_validate(classifier, x, y, scoring=scoring, cv=kf)
        scores.append((np.mean(score["test_precision"]), i,j))

# report
print(sorted(scores, key=lambda z : z[0], reverse=True)[:10])


