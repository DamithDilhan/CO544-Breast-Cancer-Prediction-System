import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold , cross_validate


# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

# use Kfold
kf = KFold(n_splits=5)

# scoring
scoring = ["accuracy", "f1", "precision"]
# use svm

classifier = make_pipeline(StandardScaler(), SVC(kernel="sigmoid", gamma="auto"))

# use cross validation

scores = cross_validate(classifier, x, y, scoring=scoring, cv=kf)

# report
print(np.mean(scores["test_accuracy"]))


