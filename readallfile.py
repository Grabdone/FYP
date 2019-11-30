# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:35:32 2019

@author: Saad
"""

import os
import glob

path = "TEST/DR1/FAKS0/"
files = os.listdir(path)

for filename in glob.glob(os.path.join(path, '*.wav')):
    samplerate, data = audioBasicIO.read(filename)
    zero.append(data)