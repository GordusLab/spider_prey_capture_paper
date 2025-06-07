# -*- coding: utf-8 -*-
"""
Created on Wed Jan 05 2022

@author: Hsin-Yi

Calculating the Fourier spectrum of the web vibration
"""


def FFT_peak(freq_m1, freq_m2, fname=None):


    import numpy as np


    ### Read the npy data ###

    if fname is None or not fname.lower().endswith('.avi'):
        raise NotImplementedError("Invalid filename for preprocessing. Please select an avi video.")

    fft = np.load(fname.replace('.avi', '_fft') + '.npz')
    ff = fft['ff']
    dataFFT = fft['dataFFT']

    fft_mean = np.mean(dataFFT, axis=0)[(ff > freq_m1) & (ff < freq_m2)]
    
    fft_max = fft_mean.max()
    
    ff_temp = ff[(ff > freq_m1) & (ff < freq_m2)]
    fft_max_index = ff_temp[np.argwhere(fft_mean == fft_mean.max())]

    print('FFT maximum = ' + str(fft_max))
    print('FFT maximum frequency = ' + str(fft_max_index))

    return 