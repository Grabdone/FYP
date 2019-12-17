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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

svclassifier = SVC(kernel='linear')
rfclass = RandomForestClassifier(max_depth=2, random_state=0)
dtclass =tree.DecisionTreeClassifier()
gnb = GaussianNB()


svclassifier.fit(X_train, y_train)
rfclass.fit(X_train, y_train)
dtclass.fit(X_train, y_train)
gnb.fit(X_train, y_train)

svmpred = svclassifier.predict(X_test)
rfpred = rfclass.predict(X_test)
dtpred = dtclass.predict(X_test)
gnbpred = gnb.predict(X_test)

eclf1 = VotingClassifier(estimators=[ ('svm', svclassifier),('rf', rfclass), ('dt', dtclass), ('gnb', gnb)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
eclfpred = eclf1.predict(X_test)


print(confusion_matrix(y_test,eclfpred))
print(classification_report(y_test,eclfpred))
print(accuracy_score(y_test, eclfpred))


