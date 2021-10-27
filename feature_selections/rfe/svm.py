import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

columns = x.columns

# standardize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# use svm
classifier = SVC(kernel="linear", probability=True)

rfe = RFECV(classifier, step=1,min_features_to_select=1, cv = 10)

fit = rfe.fit(x, y)

print(fit.support_)
print(sorted(zip(fit.ranking_,columns,fit.support_)))



