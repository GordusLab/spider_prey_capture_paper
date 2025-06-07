#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:41:28 2023

@author: hsinyihung
"""


import os, glob
import imageio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

control_name = 'FFT_table_spider_control_v.csv'
fft_name = 'FFT_table_spider_prey_v.csv'
directory = '/Users/hsinyihung/Documents/PhD/JHU/GordusLab/Spider_prey_vibration/web_vibration/result/'
fname = directory+fft_name

df_control = pd.read_csv(directory+control_name, index_col=[0])
df_fft = pd.read_csv(directory+fft_name, index_col=[0])



### 0-50 HZ
f80_100 = df_control.iloc[:, 699:875]
#f80_100 = df_control.iloc[:, 1:2]
#mean_power = f80_100.mean(axis =1)
df_control = df_control.drop(df_control.columns[437:875], axis =1)
df_control = df_control.drop(columns=['fname'])
df_control = df_control.drop(df_control.columns[0:20], axis =1)
#df_control= df_control.div(mean_power, axis=0)

f80_100 = df_fft.iloc[:, 699:875]
#f80_100 = df_fft.iloc[:, 1:2]
#mean_power = f80_100.mean(axis =1)
df_fft = df_fft.drop(df_fft.columns[437:875], axis =1)
df_fft = df_fft.drop(columns=['fname'])
df_fft = df_fft.drop(df_fft.columns[0:20], axis =1)
#df_fft= df_fft.div(mean_power, axis=0)
#df_control = df_control.drop(df_control.columns[0:7], axis =1)
#df_fft = df_fft.drop(df_fft.columns[0:7], axis =1)


ff = np.array(df_fft.columns[0:len(df_fft.columns)], dtype = float)
#index = np.array(range(0,len(df2)+1), dtype = float)
index = np.array(range(0,len(df_fft)), dtype = float)


#df2 = df_fft.div(df_control, axis=0)
df2 = df_fft.sub(df_control, axis=0)

plt.style.use('dark_background')
fig,ax = plt.subplots()
im = ax.pcolormesh(ff, index, df2, vmin = 0, vmax = 3000)
fig.colorbar(im)
ax.set_yticks(np.arange(0,len(index)))
ax.set_yticklabels(np.arange(1,len(index)+1))
plt.savefig(fname.replace(".csv", "_cut2.5hz_sub_heatmap_black.png"), dpi = 300)

plt.style.use('default')
plt.figure()
fig,ax = plt.subplots()
im = ax.pcolormesh(ff, index, df2, vmin = 0, vmax = 3000)
fig.colorbar(im)
ax.set_yticks(np.arange(0,len(index)))
ax.set_yticklabels(np.arange(1,len(index)+1))
plt.savefig(fname.replace(".csv", "_cut2.5hz_sub_heatmap.svg"))

#plt.style.use('dark_background')

#fig,ax = plt.subplots()
#im = ax.plot(ff[(ff>1) & (ff<49.5)],  df2.iloc[0][(ff>1) & (ff<49.5)])
#plt.savefig('1101_spider_piezo_5hz_75_182_with_pulses_2sdelayed_fftsub.png', dpi = 300)
