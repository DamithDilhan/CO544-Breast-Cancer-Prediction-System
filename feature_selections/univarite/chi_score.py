import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

np.set_printoptions(precision=3)

test = SelectKBest(score_func=chi2, k = 9)
fit = test.fit(x, y)

zipped = zip(fit.scores_, x.columns)
print(sorted(zipped, reverse=True))


