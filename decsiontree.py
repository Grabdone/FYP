# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:56:53 2019

@author: Saad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


bankdata2 = pd.read_csv("databank.csv")
bankdata2.shape
X = bankdata2.drop('Gender', axis=1)
y = bankdata2['Gender']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
from sklearn import tree
clf =tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred,normalize=True))


