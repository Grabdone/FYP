# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:18:06 2019

@author: Saad
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix




bankdata2 = pd.read_csv("training.csv")
bankdata2.shape
X = bankdata2.drop('Gender', axis=1)
y = bankdata2['Gender']

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = SVC(kernel='linear')
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
from matplotlib import pyplot as plt
plt.figure(figsize=(10,10))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(34).plot(kind='bar')
plt.show()
