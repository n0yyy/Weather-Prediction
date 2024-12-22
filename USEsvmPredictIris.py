# identify the weather based on 4 input features using SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

irisdata = pd.read_csv ('SortedweatherNum.csv')
X = irisdata.drop('weather', axis=1)
y = irisdata['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size = 0.30)

#Training SVM kernel 'linear','poly', 'rbf'
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#Make prediction
y_pred = svclassifier.predict(X_test)

#Evaluate SVM accuracy performance
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred, zero_division=0))
