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


# Now compute relative joint position w.r.t. connecting joints
#datarelRel = datarel.copy()[:, :, 0:2]
#for j in JOINT_PARTNERS.keys():
#    datarelRel[:, j, :] -= datarel[:, JOINT_PARTNERS[j], 0:2]

    # Rotate such that the joint partner is fixed in orientation
#    p2 = datarel[:, JOINT_PARTNERS[j], 0:2]
#    p1 = datarel[:, JOINT_PARTNERS[JOINT_PARTNERS[j]], 0:2]
#    v = p2 - p1
#    theta2 = np.arctan2(v[:, 0], v[:, 1])
#    applyRotation(theta2, datarelRel[:, j, :])

def STFT(fname, filename, dirOut, joints):
    os.makedirs(os.path.join(dirOut, 'wavelet/'), exist_ok=True)
    s = ''.join(str(x) for x in joints)
    fnameOut = dirOut +'/wavelet/'+fname+s+'_stft.mp4'
    os.makedirs(os.path.join(dirOut, 'croprot/'), exist_ok=True)
    fname = os.path.join(dirOut,
                         'croprot/'+fname+'.npy')


    data = np.load(fname)[:, :, 0:2]
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
    datarelRel = np.copy(datarel)
    x_center = np.mean(datarelRel[:,:,0], axis=0)
    datarelRel[:,:,0] = datarelRel[:,:,0]-x_center
    y_centroid = np.mean(datarelRel[:,:,1], axis=1)
    temp = datarelRel[:,:,1].T-y_centroid
    datarelRel[:,:,1] = temp.T

    datarel = datarelRel

    dataFFT = np.abs(scipy.fft.fft(datarelRel, axis =0))

    ff = np.fft.fftfreq(dataFFT.shape[0], 0.01)

    f_spec = np.zeros((100, len(range(50, data.shape[0]-50, 10)), 20,2))
    c=0
    for t in range(50, (data.shape[0] - 50), 10):
        dataSTFFT = np.abs(scipy.fft.fft(datarelRel[(t-50):(t+50),:,:], axis =0))
        f_spec[:,c,:,:] = dataSTFFT
        c+=1


    ff = np.fft.fftfreq(dataSTFFT.shape[0], 0.01)
    t = [i for i in range(50, (data.shape[0] - 50), 10)]
    t = np.array(t)
    f_idx = np.where((ff >= 0) & (ff <= 50))


    x_spectrum = f_spec[f_idx,:,:,0][0]
    y_spectrum = f_spec[f_idx,:,:,1][0]
    for i in range(x_spectrum.shape[2]):
        if i==0:
            x_f = x_spectrum[:,:,i]
            y_f = y_spectrum[:,:,i]
        else:
            x_f = np.vstack((x_f,x_spectrum[:,:,i]))
            y_f = np.vstack((y_f,y_spectrum[:,:,i]))
    
    


    vid_dir = filename.split('/aligned')[0] + filename.split('/aligned')[1].split('_croprotaligned')[0]
    vid_name = glob.glob(vid_dir + '*_labeled.mp4')[0]

    vid = imageio.get_reader(vid_name, 'ffmpeg')
    cap = cv2.VideoCapture(vid_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        try:
            ret, buf[fc] = cap.read()
            buf[fc] = cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
            fc += 1
        except:
            pass
    cap.release()

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=20, metadata=metadata)





    ff_z = np.array(list(np.arange(0,50*len(joints))))

    fig = plt.figure()
    ##Plot Tibia-Femur joint
    ax2 = plt.subplot(221)
    lm2 = ax2.pcolormesh(t, ff_z, x_f[ff_z,:], vmax = 500)
    # ax2.xaxis.set_ticks([])
    plt.title("Wavelet transform: x coordinates")
    # plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
    ln2 = ax2.axvline(0, color='red')

    ax3 = plt.subplot(222)
    lm3 = ax3.pcolormesh(t, ff_z, y_f[ff_z,:], vmax = 500)# ax3.xaxis.set_ticks([])
    plt.title("Wavelet transform: y coordinates")
    ln3 = ax3.axvline(0, color='red')
    
    ax1 = plt.subplot(212)
    im = ax1.imshow(buf[0])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    fig.tight_layout()

    
    
    with writer.saving(fig, fnameOut, 300):
        for i in range(frameCount):
        
            # ln.set_xdata(3000+x)
            ln2.set_xdata(i)
            ln3.set_xdata(i)
            im.set_data(buf[i])

            writer.grab_frame()
        
    
