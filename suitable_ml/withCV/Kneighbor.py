import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_validate


# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

# use Kfold
kf = ShuffleSplit(train_size=None,test_size=.3,n_splits=11, random_state=1)

# scoring
scoring = ["accuracy", "f1", "precision"]
# use svm

classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=19))

# use cross validation

scores = cross_validate(classifier, x, y, scoring=scoring, cv=kf)

# report
print(np.mean(scores["test_accuracy"]))


