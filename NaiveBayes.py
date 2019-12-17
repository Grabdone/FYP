# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:08:44 2019

@author: Saad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


bankdata2 = pd.read_csv("databank.csv")
bankdata2.shape
X = bankdata2.drop('Gender', axis=1)
y = bankdata2['Gender']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))