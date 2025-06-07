# -*- coding: utf-8 -*-
"""
Created on Wed Jan 05 2022

@author: Hsin-Yi

Calculating the Area under the FFT spectrum between two frequencies across the web.
"""
from loadAnnotations import *
from preprocessing_avitonpy import *

def spatial_AUC_web_vibration_sliding_window(fname, freq_m1, freq_m2, npydata=None):

    import skimage.draw, numpy as np
    from skimage.morphology import square
    import os, glob, scipy
    import math
    import cv2
    from cv2 import VideoWriter_fourcc
    from scipy import stats


    ### Check if the FFT analyis has been saved ###
    if os.path.exists(fname.replace('.avi', '_')+str(freq_m1)+'-'+str(freq_m2)+'hz_auc.npz'):
        auc = np.load(fname.replace('.avi', '_')+str(freq_m1)+'-'+str(freq_m2)+'hz_auc.npz')
        auc_data  = auc['AUC']
        print('The AUC data already exist.')
        return auc_data

    ### Read the npy data###

    if fname is None or not fname.lower().endswith('.avi'):
        raise NotImplementedError("Invalid filename for preprocessing. Please select an avi video.")

    filename = fname.replace(".avi", ".xyt") + '.npy.txt'

    if npydata is not None:
        data = npydata
    elif os.path.exists(fname.replace(".avi", ".xyt") + '.npy'):
        data = np.load(fname.replace(".avi", ".xyt") + '.npy')
    else:
        data = preprocessing_avitonpy(fname)

    ### Get web by annotaation ###
    if os.path.exists(filename):
        annotations = loadAnnotations(filename)
        lines = annotations[0][3]
        points = annotations[0][1]

        webmask = np.full((data.shape[0], data.shape[1]), False, dtype=np.bool)
        for line in lines:
            rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
            webmask[rr, cc] = True

        for point in points:
            webmask[point[0], point[1]] = True
        webmask_origin = np.copy(webmask)
        webmask = skimage.morphology.dilation(webmask, square(3))
    elif os.path.exists(fname.replace(".avi", "_mask.npy")):
        webmask = np.load(fname.replace(".avi", "_mask.npy"))
        webmask_origin = np.copy(webmask)
    elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300_modified.npy")):
        webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300_modified.npy"))
        webmask_origin = np.copy(webmask)
    elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300.npy")):
        webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300.npy"))
        webmask = webmask.astype(bool)
        webmask_origin = np.copy(webmask)

    else:
        raise NotImplementedError("No annotation file found.")


    res = np.where(webmask == True)
    res_origin = np.where(webmask_origin == True)

    img = (webmask).astype(float)
    img = img * 255
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = 0.8
    beta = (1.0 - alpha);
    AUC = np.zeros((data.shape[0], data.shape[1], len(range(1000, data.shape[2], 100))))
    c = 0
    for t in range(1000, data.shape[2], 100):
        dataFFT_web = np.abs(scipy.fft.fft(data[res[0], res[1], (t - 1000):t]))
        dataFFT = np.empty((data.shape[0], data.shape[1], 1000))
        dataFFT[:] = np.nan
        dataFFT[res[0], res[1]] = dataFFT_web
        ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)
        # snr = np.zeros((data.shape[0], data.shape[1]))
        # low_freq = np.zeros((data.shape[0], data.shape[1]))
        auc_short = np.zeros((data.shape[0], data.shape[1]))
        #### This block is the code for averaging fft alone the line
        for j in range(len(res_origin[0])):
            x_idx = res_origin[0][j]
            y_idx = res_origin[1][j]
            means = np.nanmean(np.nanmean(dataFFT[(x_idx - 1): (x_idx + 2),
                                          (y_idx - 1): (y_idx + 2), :], axis=0), axis=0)
            if math.isnan(means[0]):
                continue

            idx_i = (np.abs(ff - freq_m1)).argmin()
            idx_e = (np.abs(ff - freq_m2)).argmin()

            temp = means[idx_i:idx_e]
            auc_short[(x_idx - 1): (x_idx + 2), (y_idx - 1): (y_idx + 2)] = sum(means[idx_i:idx_e])
            temp2 = list(temp)
            temp2_max = temp.max()
            temp2.remove(temp.max())
            temp2 = np.array(temp2)
            if np.isnan(temp2_max / temp2.std()):
                continue

        #### This block is the code for calculating the snr and low frequency for dilated images
        # idx_i = (np.abs(ff - (freq-100))).argmin()
        # idx_e =  (np.abs(ff - (freq+100))).argmin()
        # fft = dataFFT_web[:, idx_i:idx_e]
        # fft_max = np.amax(fft, axis =1)
        # m, n = fft.shape
        # temp = np.where(np.arange(n-1) < fft.argmax(axis=1)[:, None], fft[:, :-1], fft[:, 1:])
        # fft_var = np.var(temp, axis =1)
        # index = np.where(fft_var < (1e-10))[0]

        # snr[res[0], res[1]] = fft_max / fft_var
        # snr = np.nan_to_num(snr, 1)
        # snr[np.where(snr>1)] =1
        # low_freq[res[0], res[1]] = np.mean(dataFFT_web[:, 1:1000], axis =1)

        AUC[:, :, c] = auc_short

        c = c + 1

    np.savez(fname.replace('.avi', '_') + str(int(freq_m1)) + '-' + str(int(freq_m2)) + 'hz_auc', AUC=AUC)
    print('The AUC data has been saved.')


    auc_map = np.copy(AUC)
    auc_map[np.isnan(auc_map)] = 0
    print('AUC max = ' + str(auc_map.max()))
    if 'control' in fname:
        auc_map[np.where(auc_map>5000)]=5000
    auc_map = auc_map / auc_map.max() * 255
    auc_map = auc_map.astype(np.uint8)

    images_snr = []
    images_auc = []

    print('Create videos for the AUC map.')

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as manimation
    import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
    import imageio
    from moviepy.editor import VideoFileClip

    ## Specify ffmpeg on HsinYi's macbook pro
    #plt.rcParams['animation.ffmpeg_path'] = '/Users/hsinyihung/opt/anaconda3/bin/ffmpeg'
    ##
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    data0 = data[:, :, 0]
    writer = FFMpegWriter(fps=15, metadata=metadata)

    for j in range(0, len(range(1000, data.shape[2], 100))):
        images_auc.append(auc_map[:, :, j])

    fig = plt.figure()
    ax2 = plt.subplot(122)
    lm = ax2.imshow(images_auc[0], alpha=1, cmap='hot')
    #lm = ax2.imshow(data[:, :, 0], cmap='gray', alpha =0.2)
    ax1 = plt.subplot(121)
    im = ax1.imshow(data[:, :, 0], cmap='gray')

    x = 0
    c = 1
    with writer.saving(fig, fname.replace(".avi", "_auc_square3_webannotation_") + str(int(freq_m1)) + '-' + str(
            int(freq_m2)) + 'hz.mp4', 100):
        for i in range(len(range(1000, data.shape[2], 100)) - 1):
            x += 100
            if x > 1000:
                lm.set_data(images_auc[c])
                #lm.set_data(data[:, :, x])
                # ln.set_data(pltsnr_20[:, :, c])
                # lo.set_data(pltsnr_30[:, :, c])
                # lp.set_data(pltsnr_60[:, :, c])
                c = c + 1
            if x > data.shape[2]:
                break
            im.set_data(data[:, :, x])

            writer.grab_frame()

    return AUC