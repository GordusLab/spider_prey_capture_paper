#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:24:39 2022

@author: hsinyihung
"""
from check_model_labels import *
from croprot_videoalignment import *
import os, glob


#directory = 'C:/Users/Hsin-Yi/Documents/GitHub/DeepLabCut/DeepLabCut_Anthony/videos'
directory ='B:\HsinYi\DeepLabCut_Anthony\8videos_1400frames_relabled/videos/new_prey/'

files = glob.glob(directory+'/082823*DLC*_100000.h5')
# files = glob.glob(directory+'/*DLC*_50000.h5')
#files = glob.glob(directory+'/032822 Spider Piezo 5Hz 0 107 With Pulses 2Sdelayed-03282022162444-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000.h5')
for i in range(0,len(files)):
#for i in range(16,len(files)):
    filename = files[i]
    joints = check_model_labels(filename)
    croprot_videoalignment(filename, joints)


