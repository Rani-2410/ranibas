# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 12:45:35 2019

@author: m
"""
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
# Load the library with the iris dataset
from sklearn.datasets import load_iris
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# Load sklearn.model_selection library
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed
np.random.seed(1234)

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the top 5 rows
df.head()

# Add a new column with the species names, this is what we are going to try to predict
target = pd.DataFrame(iris.target, columns=['species'])

# View the top 5 rows
target.head(5)
target['species'].unique()

x_train,x_test,y_train,y_test=train_test_split(df, target,
                                               test_size=0.3,
                                               random_state=25)
y_train.columns
y=pd.factorize(y_train['species'])[0]
y1 = pd.factorize(y_test['species'])[0]

clf = RandomForestClassifier().fit(x_train,y)

print(accuracy_score(y,clf.predict(x_train)))
#Train_data accuracy is 0.99

print(accuracy_score(y1,clf.predict(x_test)))
#Test_data accuracy is 0.933

from sklearn.metrics import confusion_matrix
print("Confusion Matrix: ",
        confusion_matrix(y1, clf.predict(x_test)))

from sklearn.metrics import classification_report

print("Report : ",
    classification_report(y1, clf.predict(x_test)))




