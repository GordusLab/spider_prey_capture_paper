#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:21:25 2022

@author: hsinyihung
"""

import os, glob
import imageio
from moviepy.editor import VideoFileClip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fname = '/Users/hsinyihung/Documents/PhD/JHU/Gordus lab/Spider_prey_vibration/web_vibration/result/FFT_table_spider_piezo.csv'

df2 = pd.read_csv(fname, index_col=[0])
f80_100 = df2.iloc[:, 699:875]
mean_power = f80_100.mean(axis =1)

### 0-25 HZ
#df2 = df2.drop(df2.columns[218:874], axis =1)

### 0-50 HZ
df2 = df2.drop(df2.columns[436:875], axis =1)
df2 = df2.drop(columns=['fname'])
df2 = df2.drop(df2.columns[0:7], axis =1)


ff = np.array(df2.columns[0:len(df2.columns)], dtype = float)
#index = np.array(range(0,len(df2)+1), dtype = float)
index = np.array(range(0,len(df2)), dtype = float)

first_column = df2.iloc[:, 0]
#df2 = df2.div(first_column, axis=0)
df2 = df2.div(mean_power, axis=0)


fig,ax = plt.subplots()
im = ax.pcolormesh(ff, index, df2, vmin = 0, vmax = 20)
fig.colorbar(im)
ax.set_yticks(np.arange(0,len(index)))
ax.set_yticklabels(np.arange(1,len(index)+1))
plt.savefig(fname.replace(".csv", "_heatmap.png"), dpi = 300)

