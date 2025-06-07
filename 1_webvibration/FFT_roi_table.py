# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:48:40 2022

@author: Hsin-Yi
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 08:32:13 2022

@author: hsinyihung
"""

import os, glob
import imageio
import numpy as np
import pandas as pd
import scipy.interpolate as interp

### Load files
main_dir = 'Z:/HsinYi/web_vibration'
# main_dir = '/Volumes/Team Spider/HsinYi/web_vibration'
files = glob.glob(main_dir + '/*/*/*_roi_fft.npz')

for fname in files:

    data = np.load(fname)
    ff = data['ff']
    fft_fly = data['dataFFT_fly']
    fft_fly_p = data['dataFFT_fly_p']
    fft_spider = data['dataFFT_spider']
    fft_spider_p = data['dataFFT_spider_p']

    column_names = [ff[(ff > 0) & (ff < 100)]]

    ##Fly
    temp = fft_fly[(ff > 0) & (ff < 100)]
    if len(temp) < 873:
        arr2_interp = interp.interp1d(np.arange(temp.size), temp)
        arr2_stretch = arr2_interp(np.linspace(0, temp.size - 1, 873))
        temp = arr2_stretch
        arr2 = column_names[0]
        arr2_interp = interp.interp1d(np.arange(arr2.size), arr2)
        arr2_stretch = arr2_interp(np.linspace(0, arr2.size - 1, 873))
        column_names[0] = arr2_stretch
    df = pd.DataFrame(temp.reshape(1, len(temp)), columns=column_names)
    df.insert(0, "fname", fname)
    if os.path.exists(main_dir + '/FFT_roi_fly_table.csv'):
        df.to_csv(main_dir + '/temp.csv')
        df = pd.read_csv(main_dir + '/temp.csv', index_col=[0])
        df2 = pd.read_csv(main_dir + '/FFT_roi_fly_table.csv', index_col=[0])

        frames = [df, df2]
        result = pd.concat(frames)
        result.to_csv(main_dir + '/FFT_roi_fly_table.csv')
    else:
        df.to_csv(main_dir + '/FFT_roi_fly_table.csv')

    ## Spider
    temp = fft_spider[(ff > 0) & (ff < 100)]
    if len(temp) < 873:
        arr2_interp = interp.interp1d(np.arange(temp.size), temp)
        arr2_stretch = arr2_interp(np.linspace(0, temp.size - 1, 873))
        temp = arr2_stretch
        arr2 = column_names[0]
        arr2_interp = interp.interp1d(np.arange(arr2.size), arr2)
        arr2_stretch = arr2_interp(np.linspace(0, arr2.size - 1, 873))
        column_names[0] = arr2_stretch
    df = pd.DataFrame(temp.reshape(1, len(temp)), columns=column_names)
    df.insert(0, "fname", fname)
    if os.path.exists(main_dir + '/FFT_roi_spider_table.csv'):
        df.to_csv(main_dir + '/temp.csv')
        df = pd.read_csv(main_dir + '/temp.csv', index_col=[0])
        df2 = pd.read_csv(main_dir + '/FFT_roi_spider_table.csv', index_col=[0])

        frames = [df, df2]
        result = pd.concat(frames)
        result.to_csv(main_dir + '/FFT_roi_spider_table.csv')
    else:
        df.to_csv(main_dir + '/FFT_roi_spider_table.csv')

    ## Fly p
    temp = fft_fly_p[(ff > 0) & (ff < 100)]
    if len(temp) < 873:
        arr2_interp = interp.interp1d(np.arange(temp.size), temp)
        arr2_stretch = arr2_interp(np.linspace(0, temp.size - 1, 873))
        temp = arr2_stretch
        arr2 = column_names[0]
        arr2_interp = interp.interp1d(np.arange(arr2.size), arr2)
        arr2_stretch = arr2_interp(np.linspace(0, arr2.size - 1, 873))
        column_names[0] = arr2_stretch
    df = pd.DataFrame(temp.reshape(1, len(temp)), columns=column_names)
    df.insert(0, "fname", fname)
    if os.path.exists(main_dir + '/FFT_roi_fly_p_table.csv'):
        df.to_csv(main_dir + '/temp.csv')
        df = pd.read_csv(main_dir + '/temp.csv', index_col=[0])
        df2 = pd.read_csv(main_dir + '/FFT_roi_fly_p_table.csv', index_col=[0])

        frames = [df, df2]
        result = pd.concat(frames)
        result.to_csv(main_dir + '/FFT_roi_fly_p_table.csv')
    else:
        df.to_csv(main_dir + '/FFT_roi_fly_p_table.csv')



    ## Spider p
    temp = fft_spider_p[(ff > 0) & (ff < 100)]
    if len(temp) < 873:
        arr2_interp = interp.interp1d(np.arange(temp.size), temp)
        arr2_stretch = arr2_interp(np.linspace(0, temp.size - 1, 873))
        temp = arr2_stretch
        arr2 = column_names[0]
        arr2_interp = interp.interp1d(np.arange(arr2.size), arr2)
        arr2_stretch = arr2_interp(np.linspace(0, arr2.size - 1, 873))
        column_names[0] = arr2_stretch
    df = pd.DataFrame(temp.reshape(1, len(temp)), columns=column_names)
    df.insert(0, "fname", fname)
    if os.path.exists(main_dir + '/FFT_roi_spider_p_table.csv'):
        df.to_csv(main_dir + '/temp.csv')
        df = pd.read_csv(main_dir + '/temp.csv', index_col=[0])
        df2 = pd.read_csv(main_dir + '/FFT_roi_spider_p_table.csv', index_col=[0])

        frames = [df, df2]
        result = pd.concat(frames)
        result.to_csv(main_dir + '/FFT_roi_spider_p_table.csv')
    else:
        df.to_csv(main_dir + '/FFT_roi_spider_p_table.csv')