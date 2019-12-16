# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:09:34 2019

@author: Saad
"""

from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
from mlxtend.plotting import plot_decision_regions
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

df=pd.DataFrame()

[Fs, xx] = audioBasicIO.readAudioFile("TEST/DR1/FAKS0/SA1.WAV.wav");
plt.plot(xx)
F, f_names = audioFeatureExtraction.stFeatureExtraction(xx, Fs, 0.050*Fs, 0.025*Fs);
f_names.append("Gender")
zz=pd.DataFrame(F.tolist())
mean=zz.mean(axis=1)
gender=pd.Series("male")
mean=mean.append(gender, ignore_index=True)
mean=mean.to_frame()
mean=mean.transpose()
df=df.append(mean)
df.columns=f_names

svclassifier = SVC(kernel='linear')
svclassifier.fit(X, y)

X_test = df.drop('Gender', axis=1)
y_test = df['Gender']

y_pred = svclassifier.predict(X_test)


print("X=%s, Predicted=%s" % (y_test[0],y_pred))


