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

directory="TEST/DR1/FAKS0/"
df=pd.DataFrame()
i=0
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        print(os.path.join(directory, filename))
        [Fs, x] = audioBasicIO.readAudioFile(os.path.join(directory, filename));
        F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
        zz=pd.DataFrame(F.tolist())
        mean=zz.mean(axis=1)
        mean=mean.to_frame()
        mean=mean.transpose()
        df=df.append(mean)
        i=+1
        continue
    else:
        continue

df.columns=f_names
    
        
        
        
        
#        data = pd.Series(index=f_names)
#        data.append(mean,ignore_index=True)
    
        
       
        
        


#[Fs, x] = audioBasicIO.readAudioFile("TEST/DR1/FAKS0/SA1.WAV.wav");
#F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
#zz=pd.DataFrame(F.tolist())
#zz=zz.mean(axis=1)


#kk=os.system('python audioAnalysis.py  featureExtractionDir -i TEST/DR1/FAKS0/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050')
#
#import subprocess
#result = subprocess.run(['python audioAnalysis.py  featureExtractionDir -i TEST/DR1/FAKS0/SA1.WAV.wav -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050'], stdout=subprocess.PIPE)