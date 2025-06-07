import glob
import numpy as np
import os, matplotlib.pyplot as plt, scipy
files = glob.glob('B:\HsinYi\DeepLabCut_Anthony\8videos_1400frames_relabled/videos/2024dlc_model/aligned\wavelet/111021 Spider Prey*_nonormalized_*.npz')
cut = 550

for i in range(len(files)):
    data = np.load(files[i])
    wavelet_x = data['arr_0']
    wavelet_y = data['arr_1']
    ff = data['arr_2']
    nframe = wavelet_x.shape[0]
    #endframe = nframe
    wavelet_x = wavelet_x[cut:nframe,:]
    wavelet_y = wavelet_y[cut:nframe, :]
    np.savez(files[i], wavelet_x, wavelet_y, ff)

    fig = plt.figure()
    ##Plot Tibia-Femur joint
    ax2 = plt.subplot(221)
    lm2 = ax2.pcolormesh(np.array(list(np.arange(0,wavelet_x.shape[0]))), np.array(list(np.arange(0,50*5))), wavelet_x.T, vmax = 0.5)
    # ax2.xaxis.set_ticks([])
    plt.title("Wavelet transform: x coordinates")
    # plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
    ln2 = ax2.axvline(0, color='red')

    ax3 = plt.subplot(222)
    lm3 = ax3.pcolormesh(np.array(list(np.arange(0,wavelet_y.shape[0]))), np.array(list(np.arange(0,50*5))), wavelet_y.T, vmax = 0.5)
    plt.title("Wavelet transform: y coordinates")
    ln3 = ax3.axvline(0, color='red')

    ax1 = plt.subplot(212)
    #im = ax1.imshow(buf[0])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    fig.tight_layout()
    plt.savefig(files[i].replace('.npz','.png'))