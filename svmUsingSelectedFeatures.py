# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:51:48 2019

@author: Saad
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


bankdata2 = pd.read_csv("databank.csv")
X = bankdata2[['mfcc_13','mfcc_12','mfcc_10','spectral_flux']]
y = bankdata2['Gender']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


y_pred = svclassifier.predict(X_test)


from sklearn.metrics import accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred,normalize=True))


