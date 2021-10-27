import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

# split train and test

x_train , x_test , y_train , y_test = train_test_split(x, y, train_size=0.8)

# use svm

classifier = make_pipeline(StandardScaler(), SVC(kernel="sigmoid", C=0.6, gamma="auto"))

classifier.fit(x_train, y_train)

# report
report = classification_report(y_test, classifier.predict(x_test))

print(report)

