#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:22:15 2022

@author: hsinyihung
"""

import os, glob, numpy as np, matplotlib.pyplot as plt, skimage, skimage.draw, scipy.io
import wavelet_coxareference_sidecamera, STFT_sidecamera
#import centerjointcoordinate_sidecamera


directory = 'B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/aligned/croprot/coxareference'
#directory = 'B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/aligned/newvideos'

#os.makedirs(os.path.join(directory, '/croprot/'), exist_ok=True)
files = glob.glob(directory+'/*.npy')

#fname = '011822  Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed-01182022141158-0000-1_croprotaligned'
#filename = '/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/aligned/'+fname +'.npy'
for n in range(len(files)):
    filename = files[n]
    fname = files[n].split('/coxareference\\')[1]
    #fname = files[n].split('/newvideos\\')[1]
    fname = fname.split('.npy')[0]
    dirOut = directory


    joints = np.array(range(0,4))
    wavelet_coxareference_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)
    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)
    joints = np.array(range(4, 8))
    wavelet_coxareference_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)
    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)
    joints = np.array(range(8, 12))
    wavelet_coxareference_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)
    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)
    joints = np.array(range(12, 16))
    wavelet_coxareference_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)

    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)