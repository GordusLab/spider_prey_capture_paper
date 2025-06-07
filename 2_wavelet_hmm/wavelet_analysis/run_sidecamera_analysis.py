#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:22:15 2022

@author: hsinyihung
"""

import os, glob, numpy as np, matplotlib.pyplot as plt, skimage, skimage.draw, scipy.io
import wavelet_sidecamera, STFT_sidecamera


directory = 'B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/2023dlc_model_nocut/aligned'
#directory = 'B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/newvideo_repeatedpiezo/aligned'

os.makedirs(os.path.join(directory, '/croprot/'), exist_ok=True)
files = glob.glob(directory+'/*.npy')

#fname = '011822  Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed-01182022141158-0000-1_croprotaligned'
#filename = '/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/aligned/'+fname +'.npy'
for n in range(len(files)):
    filename = files[n]
    fname = files[n].split('/aligned\\')[1]
    #fname = files[n].split('/newvideos\\')[1]
    fname = fname.split('.npy')[0]
    joint_data = np.load(filename)
    joints = np.delete(joint_data, range(2, joint_data.shape[0], 3), axis=0)
    joints = np.transpose(joints)

    arrData = np.zeros((joints.shape[0], 20, 3))
    for i in range(0,int(joints.shape[1]/2)):
        arrData[:,i,0] = joints[:,i*2]
        arrData[:,i,1] = joints[:,i*2+1]
    # Save
    #dirOut = '/Users/hsinyihung/Documents/PhD/JHU/Gordus lab/Spider_prey_vibration/behavioral_motifs_wavelet/8videos_1400frames_relabled/'+ fname.split('_croprotaligned')[0] +'/'
    dirOut = directory

    np.save(dirOut+'/croprot/'+fname+'.npy',arrData )
    #np.save('C:/Users/Gordus_Lab/Desktop/croprot/'+fname+'.npy',arrData)

    joints = np.array(range(0,5))
    wavelet_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)
    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)
    joints = np.array(range(5, 10))
    wavelet_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)
    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)
    joints = np.array(range(10, 15))
    wavelet_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)
    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)
    joints = np.array(range(15, 20))
    wavelet_sidecamera.Wavelet_transform(fname, filename, dirOut, joints)

    #STFT_sidecamera.STFT(fname, filename, dirOut, joints)