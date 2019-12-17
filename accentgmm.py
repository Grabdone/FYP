# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:31:49 2019

@author: Saad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


bankdata2 = pd.read_csv("DatabankWithDailect.csv")
bankdata2.shape
X = bankdata2.drop('Dialect', axis=1)
y = bankdata2['Dialect']
y2 = bankdata2['Dialect']



from sklearn import mixture
g = mixture.GaussianMixture(n_components=2).fit(X)

labels=g.predict(X)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)

print(confusion_matrix(y,labels))
print(classification_report(y,labels))
print(accuracy_score(y,labels))

plt.scatter(X['mfcc_12'],X['mfcc_13'] , c=labels, s=10, cmap='viridis');