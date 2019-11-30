# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:44:19 2019

@author: Saad
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:21:00 2019

@author: Saad
"""

import pandas as pd
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
import os

Start="TEST/"
df=pd.DataFrame()
for DR in os.listdir(Start):
    a=os.path.join(Start, DR).replace("\\","/")
    for DR1 in os.listdir(a):
        if DR1[0]=='F':
            sex="female"
        else:
            sex="male"
            
        directory=os.path.join(a, DR1).replace("\\","/")
        i=0
        for filename in os.listdir(directory):
            if filename.endswith(".wav"): 
                print(os.path.join(directory, filename))
                [Fs, x] = audioBasicIO.readAudioFile(os.path.join(directory, filename));
                F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
                f_names.append("Gender")
                zz=pd.DataFrame(F.tolist())
                mean=zz.mean(axis=1)
                gender=pd.Series(sex)
                mean=mean.append(gender, ignore_index=True)
                mean=mean.to_frame()
                mean=mean.transpose()
                df=df.append(mean)
                i=+1
                continue
            else:
                continue
        
df.columns=f_names
df.to_csv("testing.csv")

            
            
        
   