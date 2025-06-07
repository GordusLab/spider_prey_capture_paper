#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:11:51 2022

@author: hsinyihung
"""
import os
import numpy as np
from numba import jit
import os, matplotlib.pyplot as plt, scipy
import glob
import matplotlib.animation as manimation
import imageio
from moviepy.editor import VideoFileClip
import cv2
from scipy import stats
from scipy import fft




@jit(nopython=True, nogil=True)
def applyRotationAlongAxis(R, X):
    """
    This helper function applies a rotation matrix to every <X, Y> position tuple in a Nx2 matrix.
    Note: Numba JIT leads to a ~6-fold speed improvement.
    """
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j, 0:2] = R[:, :, i] @ X[i, j, 0:2]


@jit(nopython=True, nogil=True)
def applyRotationAlongAxis1d(R, X):
    """
    This helper function applies a rotation matrix to every <X, Y> position tuple in a Nx2 matrix.
    Note: Numba JIT leads to a ~6-fold speed improvement.
    """
    for i in range(X.shape[0]):
        X[i, 0:2] = R[:, :, i] @ X[i, 0:2]


def applyRotation(theta, X):
    # Create rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Perform rotation (Takes on the order of 15-30 seconds for most datasets.)
    if X.ndim == 3:
        applyRotationAlongAxis(R, X)
    elif X.ndim == 2:
        applyRotationAlongAxis1d(R, X)
def morletConjFT(w, omega0):
    return np.power(np.pi, -1 / 4) * np.exp(-0.5 * np.power(w - omega0, 2));
   
def WaveletCalc(projections, numModes=None, parameters=None):
    # ...
    d1, d2 = projections.shape
    if d2 > d1:
        projections = projections.T

    # ...
    L = projections[1, :].shape[0]
    if numModes is None:
        numModes = L;
    else:
        if numModes > L:
            numModes = L

    # Extract parameters
    omega0 = parameters['omega0']
    numPeriods = parameters['numPeriods']
    dt = 1 / parameters['samplingFreq']
    minT = 1 / parameters['maxF']
    maxT = 1 / parameters['minF']

    Ts = minT * np.power(2, np.arange(0, numPeriods) * np.log(maxT / minT) / (np.log(2) * (numPeriods - 1)))
    f = np.flip(1 / Ts)
    N = projections[:, 0].shape[0]

    if parameters['stack']:
        amplitudes = np.zeros((N, numModes * numPeriods))
        for i in range(numModes):
            temp, W = fastWavelet_morlet_convolution_parallel(
                projections[:, i], f, omega0, dt)
            temp = np.fliplr(temp)
            temp = temp / np.max(temp)
            # import pdb; pdb.set_trace()
            amplitudes[:, np.arange(i * numPeriods, (i + 1) * numPeriods, dtype=int)] = temp.T
    else:
        raise NotImplementedError("Only 'stack' mode is supported.")
        # amplitudes = zeros(N,numModes,numPeriods);
        # for i=1:numModes:
        #    temp = ...
        #        fastWavelet_morlet_convolution_parallel_ag(...
        #        projections(:,i),f,omega0,dt)';
        #    temp = temp./max(temp(:));
        #    amplitudes(:,i,:) = temp;

    return amplitudes, f
    

def fastWavelet_morlet_convolution_parallel(x, f, omega0, dt):
    N = x.shape[0]
    L = f.shape[0]
    amp = np.zeros((L, N))

    test = None
    if np.mod(N, 2) == 1:
        x = np.append(x, [0])
        N = N + 1;
        test = True
    else:
        test = False

    if len(x.shape) == 1:
        x = np.asmatrix(x)

    if x.shape[1] == 1:
        x = x.T

    x = np.hstack((np.zeros((1, int(N / 2))), x, np.zeros((1, int(N / 2)))))
    M = N
    N = x.shape[1]

    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
    Omegavals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * dt)

    xHat = np.fft.fft(x)
    xHat = np.fft.fftshift(xHat)

    idx = None
    if test:
        idx = np.arange(M / 2, M / 2 + M - 1, dtype=int)
    else:
        idx = np.arange(M / 2, M / 2 + M, dtype=int)

    returnW = True

    test2 = None
    if returnW:
        W = np.zeros(amp.shape);
        test2 = True;
    else:
        test2 = False;

    for i in range(L):
        m = morletConjFT(- Omegavals * scales[i], omega0)
        q = np.fft.ifft(m * xHat) * np.sqrt(scales[i])
        q = q[0, idx]
        amp[i, :] = np.abs(q) * np.power(np.pi, -0.25) * \
                    np.exp(0.25 * np.power(omega0 - np.sqrt(omega0 ** 2 + 2), 2)) / np.sqrt(2 * scales[i])

        if returnW:
            W[i, :] = q

    return amp, W
    
def Wavelet_transform(fname, filename, dirOut, joints):
    os.makedirs(os.path.join(dirOut, 'wavelet/'), exist_ok=True)
    fnameOut = dirOut +'/wavelet/wavelet.mp4'
    os.makedirs(os.path.join(dirOut, 'croprot/'), exist_ok=True)
    dataname = os.path.join(dirOut,
                         'croprot/'+fname+'.npy')


    data = np.load(dataname)[:, :, 0:2]
    datarel = data


    JOINT_PARTNERS = {
    0: 1, 1: 2,
    2: 3, 3: 4,
    4: 3,
    5: 6, 6: 7,
    7: 8, 8: 9,
    9: 8,
    10: 11, 11: 12,
    12: 13, 13: 14,
    14: 13,
    15: 16, 16: 17,
    17: 18, 18: 19,
    19: 18,
    }

    JOINTS_LEGS = list(range(0, 20))
    legvars = {}
    legvars['samplingFreq'] = 100;  # fps
    legvars['omega0'] = 5;
    legvars['numPeriods'] = 50;
    legvars['maxF'] = legvars['samplingFreq'] / 2;
    legvars['minF'] = 0.1;  # hz
    legvars['stack'] = True;
    legvars['numProcessors'] = 4;



    datarelRel = np.copy(datarel)
    x_center = np.mean(datarelRel[:,:,0], axis=0)
    datarelRel[:,:,0] = datarelRel[:,:,0]-x_center
    y_centroid = np.mean(datarelRel[:,:,1], axis=1)
    temp = datarelRel[:,:,1].T-y_centroid
    datarelRel[:,:,1] = temp.T

    datarel = datarelRel

       
    ### Wavelet trasnform
    dataLeg_x = np.zeros((datarelRel.shape[0], len(joints)))
    dataLeg_y = np.zeros((datarelRel.shape[0], len(joints)))
    for i in range(len(joints)):
        dataLeg_x[:,i] = datarelRel[:,i,0]
        dataLeg_y[:, i] = datarelRel[:,i,1]
    
    
    amplitudes_x, f = WaveletCalc(dataLeg_x, dataLeg_x.shape[1], parameters=legvars)
    amplitudes_y, f = WaveletCalc(dataLeg_y, dataLeg_y.shape[1], parameters=legvars)

    np.savez(dirOut +'/wavelet/wavelet.npz', amplitudes_x, amplitudes_y, f )


    #vid_dir = filename.split('/aligned')[0] + filename.split('/aligned')[1].split('_croprotaligned')[0]
    #vid_name = glob.glob(vid_dir + '*_labeled.mp4')[0]

    #vid = imageio.get_reader(vid_name, 'ffmpeg')
    #cap = cv2.VideoCapture(vid_name)
    #frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    #fc = 0
    #ret = True

    #while (fc < frameCount and ret):
    #    try:
    #        ret, buf[fc] = cap.read()
    #        buf[fc] = cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
    #        fc += 1
    #    except:
    #        pass
    #cap.release()

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=20, metadata=metadata)


    fig = plt.figure()
    ##Plot Tibia-Femur joint
    ax2 = plt.subplot(221)
    lm2 = ax2.pcolormesh(np.array(list(np.arange(0,amplitudes_x.shape[0]))), np.array(list(np.arange(0,50*len(joints)))), amplitudes_x.T, vmax = 0.5)
    # ax2.xaxis.set_ticks([])
    plt.title("Wavelet transform: x coordinates")
    # plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
    ln2 = ax2.axvline(0, color='red')

    ax3 = plt.subplot(222)
    lm3 = ax3.pcolormesh(np.array(list(np.arange(0,amplitudes_y.shape[0]))), np.array(list(np.arange(0,50*len(joints)))), amplitudes_y.T, vmax = 0.5)
    plt.title("Wavelet transform: y coordinates")
    ln3 = ax3.axvline(0, color='red')

    #ax1 = plt.subplot(212)
    #im = ax1.imshow(buf[0])
    #ax1.xaxis.set_visible(False)
    #ax1.yaxis.set_visible(False)
    fig.tight_layout()

    #with writer.saving(fig, fnameOut, 300):
    #    for i in range(frameCount):
        
            # ln.set_xdata(3000+x)
    #        ln2.set_xdata(i)
    #        ln3.set_xdata(i)
            #im.set_data(buf[i])

    #       writer.grab_frame()
        