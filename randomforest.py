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
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(train, train_labels)
#from sklearn import tree
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
#X_train, y_train = make_classification(n_samples=1000, n_features=34,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
#clf = RandomForestClassifier(max_depth=2, random_state=0)
#
#clf.fit(X_train, y_train)
#
#print(clf.feature_importances_)
#print(clf.predict(X_test))
#y_pred = clf.predict(X_test)
#
#from sklearn.metrics import accuracy_score
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print("accuracy:",accuracy_score(y_test,y_pred,normalize=True))
#

