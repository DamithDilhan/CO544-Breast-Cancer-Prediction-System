import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV 

# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

# use Kfold
kf = KFold(n_splits=10)

# scoring
scoring = ["accuracy", "f1", "precision"]

# use logistic regression

classifier =RandomForestClassifier(random_state=0)


rfe = RFECV(classifier,min_features_to_select=1, cv = kf)

fit = rfe.fit(x, y)

print(fit.support_)
print(sorted(zip(fit.ranking_, x.columns)))
