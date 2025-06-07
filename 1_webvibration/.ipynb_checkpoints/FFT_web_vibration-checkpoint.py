# -*- coding: utf-8 -*-
"""
Created on Wed Jan 05 2022

@author: Hsin-Yi

Calculating the Fourier spectrum of the web vibration
"""
from loadAnnotations import *
from preprocessing_avitonpy import *

def FFT_web_vibration(fname=None, npydata = None):
    import os, matplotlib.pyplot as plt, scipy
    import cv2
    import skimage.draw, numpy as np
    from skimage.morphology import square

    ### Check if the FFT analyis has been saved ###
    if os.path.exists(fname.replace('.avi', '_fft')+'.npz'):
        fft = np.load(fname.replace('.avi', '_fft')+'.npz')
        ff = fft['ff']
        dataFFT = fft['dataFFT']
        print('The FFT data already exist.')
        return ff, dataFFT

    ### Read the npy data ###

    if fname is None or not fname.lower().endswith('.avi'):
        raise NotImplementedError("Invalid filename for preprocessing. Please select an avi video.")

    filename = fname.replace(".avi", ".xyt") + '.npy.txt'
    fnameFFT = fname + '.fft.npy'

    if npydata is not None:
        data = npydata
    elif os.path.exists(fname.replace(".avi", ".xyt") + '.npy'):
        data = np.load(fname.replace(".avi", ".xyt") + '.npy')
    else:
        data = preprocessing_avitonpy(fname)



    ### Exclude missing frames ###
    #data = data[:, :, :-1]

    #kernel = np.ones((3,3),np.uint8)
    #for i in range(0, data.shape[2], 500):
    #   erosion = cv2.erode(data[:, :, i:(i+500)],kernel,iterations = 1)
    #   dilation = cv2.dilate(erosion, kernel,iterations = 1)
       #data[:, :, i:(i+500) ] = data[:, :, i:(i+500) ]- dilation
    #   data[:, :, i:(i+500) ] =  dilation

    ### Substract the spider/flies/stabalimentum ###
    #limb_threshold = 150
    #data_copy = np.copy(data)
    #kernel = np.ones((3,3),np.uint8)
    #for i in range(0, data.shape[2]):
    #    erosion = cv2.erode(data[:, :, i],kernel,iterations = 1)
    #    dilation = cv2.dilate(erosion, kernel,iterations = 1)
    #    data[:, :, i] = data[:, :, i]- dilation
    #    temp_mask = data[:,:,i]>limb_threshold
    #    dilation2 = cv2.dilate(dilation, kernel,iterations = 10)
    #    temp_mask2 = (dilation2>100) & (temp_mask==True)
    #    temp = data[:,:,i]
    #    temp[np.where(temp_mask2 ==True)]=0
    #    data[:,:,i] = temp
    #   #data[:, :, i:(i+500) ] =  dilation

    ### Get web by threshold  ###
    # threshold = 20
    # web_idx = data[:, :, 0]> threshold
    # res = np.where(web_idx == True)

    ### Get web by annotaation ###
    if os.path.exists(filename):
        annotations = loadAnnotations(filename)

        lines = annotations[0][3]
        points = annotations[0][1]

        webmask = np.full((np.size(data, 0), np.size(data, 1)), False, dtype=np.bool_)
        for line in lines:
            rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
            # idx1 = np.argwhere(rr>=1024)
            # idx2 = np.argwhere(cc>=1024)
            # if np.size(idx2)==0:
            #    idx = idx1
            # else:
            #    idx = idx1 or idx2
            # cc  = np.delete(cc, idx)
            # rr  = np.delete(rr, idx)
            webmask[rr, cc] = True

        for point in points:
            if point[0] >= 1024 or point[1] >= 1024:
                continue
            webmask[point[0], point[1]] = True

        webmask = skimage.morphology.dilation(webmask, square(3))
    elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300_modified.npy")):
        webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300_modified.npy"))
    elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300.npy")):
        webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300.npy"))
    elif os.path.exists(fname.replace(".avi", "_mask.npy")):
        webmask = np.load(fname.replace(".avi", "_mask.npy"))

    else:
        raise NotImplementedError("No annotation file found.")

    # webmask[0:800, :]= False
    # webmask[800:1280, 0:500]= False
    res = np.where(webmask == True)

    ### Apply FFT to data
    dataFFT = np.abs(scipy.fft.fft(data[res[0], res[1], :]))


    ### Plot and save the result
    ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
    plt.figure()
    plt.plot(ff[ff > 0], np.mean(dataFFT, axis=0)[ff > 0])
    plt.savefig(fname.replace('.avi', '_fft.png'))
    plt.figure()
    plt.plot(ff[(ff > 0) & (ff < 100)], np.mean(dataFFT, axis=0)[(ff > 0) & (ff < 100)])
    plt.savefig(fname.replace('.avi', '_2_fft.png'))
    plt.figure()
    plt.plot(ff[(ff > 1) & (ff < 100)], np.mean(dataFFT, axis=0)[(ff > 1) & (ff < 100)])
    plt.savefig(fname.replace('.avi', '_3_fft.png'))

    np.savez(fname.replace('.avi', '_fft'), ff=ff, dataFFT=np.abs(scipy.fft.fft(data[res[0], res[1], :])))
    print('The FFT data have been saved.')

    return ff, dataFFT

    