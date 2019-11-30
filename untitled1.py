# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:02:22 2019

@author: Saad
"""

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal



[Fs, xx] = audioBasicIO.readAudioFile("saad.wav");
x=stereo_to_mono(xx)
plt.plot(xx)
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
#plt.subplot(2,2,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]); 
#plt.subplot(2,2,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]);
#plt.subplot(2,2,3); plt.plot(F[2,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[2]);
#plt.subplot(2,2,4); plt.plot(F[3,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[3]);
#plt.subplot(2,2,1); plt.plot(F[21,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[21]);
#plt.subplot(2,2,2); plt.plot(F[22,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[22]); plt.show()
plt.subplot(1,1,1); plt.plot(F[7,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[7]);plt.show()


means=np.mean(F[0,:]);

[Fs2, x2] = audioBasicIO.readAudioFile("TEST/DR1/MDAB0/SA1.WAV.wav");
plt.plot(x2)
K, f_names = audioFeatureExtraction.stFeatureExtraction(x2, Fs2, 0.050*Fs, 0.025*Fs);
#plt.subplot(2,2,1); plt.plot(K[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]); 
#plt.subplot(2,2,2); plt.plot(K[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]);
#plt.subplot(2,2,3); plt.plot(K[2,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[2]);
#plt.subplot(2,2,4); plt.plot(K[3,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[3]); 
#plt.subplot(2,2,1); plt.plot(K[21,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[21]);
#plt.subplot(2,2,2); plt.plot(K[22,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[22]); plt.show()
plt.subplot(1,1,1); plt.plot(K[7,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[7]); plt.show()

#python audioAnalysis.py  featureExtractionDir -i data/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050




