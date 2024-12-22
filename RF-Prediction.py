
#Random Forest to predict weather

from sklearn import datasets
import pandas as pd
data = pd.read_csv('weatherNum.csv')
print(data)

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[[ 'precipitation' , 'temp_max', 'temp_min', 'wind' ]] # Features
y=data['weather']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 20% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=200)
#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)  #train model on training set
y_pred=clf.predict(X_test) #predict model on test set

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Display the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()
