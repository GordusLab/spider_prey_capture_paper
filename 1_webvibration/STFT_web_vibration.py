# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 2022

@author: Hsin-Yi

Calculating the Area under the FFT spectrum between two frequencies across the web.
"""

from loadAnnotations import *
from preprocessing_avitonpy import *
def STFT_web_vibration(fname, npydata=None):
    import os, numpy as np, matplotlib.pyplot as plt, scipy
    from moviepy.editor import VideoFileClip
    from scipy import stats
    from scipy import fft
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as manimation
    import skimage.draw
    from skimage.morphology import square

    window = 400
    step = 20

    ### Check if the STFT analyis has been saved ###
    if os.path.exists(fname.replace('.avi', '_stft_window400_shift20.npz')):
        spec = np.load(fname.replace('.avi', '_stft_window400_shift20.npz'))
        f_spec = spec['f_spec']
        print('The STFT data already exist.')
        return f_spec

    ### Read the npy data ###

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
        webmask_origin = webmask
        webmask = skimage.morphology.dilation(webmask, square(3))
    elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300_modified.npy")):
        webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300_modified.npy"))
        webmask_origin = webmask
    elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300.npy")):
        webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300.npy"))
        webmask_origin = webmask

    else:
        raise NotImplementedError("No annotation file found.")


    res = np.where(webmask == True)
    res_origin = np.where(webmask_origin == True)

    ### Extract the web index
    # bottom right portion of the web
    # data = data[200:600, 400:800, :]


    ### Substract the spider/flies/stabalimentum
    #kernel = np.ones((5, 5), np.uint8)
    #for i in range(0, data.shape[2], 500):
    #    erosion = cv2.erode(data[:, :, i:(i + 500)], kernel, iterations=1)
    #    dilation = cv2.dilate(erosion, kernel, iterations=1)
    #    data[:, :, i:(i + 500)] = data[:, :, i:(i + 500)] - dilation
        # data[:, :, i:(i+500) ] = dilation

    ### Extract the web index
    #threshold = 20
    # Use maximum projection as a thresold
    # maxprojection = np.amax(data, axis =2)


    ### The STFT analysis
    f_spec = np.zeros((window, len(range(window, data.shape[2], step))))
    z_spec = np.zeros((window, len(range(window, data.shape[2], step))))
    v_spec = np.zeros((window, len(range(window, data.shape[2], step))))
    # f_spec = np.zeros((1000, len(range(4000, 6000, 10))))

    c = 0
    for t in range(int(window/2), (data.shape[2] - int(window/2)), step):
        # for t in range(4000, 6000, 10):
        dataFFT = np.abs(scipy.fft.fft(data[res[0], res[1], (t - int(window/2)):(t + int(window/2))]))
        f_spec[:, c] = np.mean(dataFFT, axis=0)
        z_score = stats.zscore(dataFFT, axis=0)
        z_spec[:, c] = np.mean(np.abs(z_score), axis=0)
        v_spec[:, c] = np.var(dataFFT, axis=0) / np.mean(dataFFT, axis=0)
        # dataFFT_b = np.abs(scipy.fft(baseline_data[res[0], res[1], (t-1000):t]))
        # f_spec[:,c] = np.mean(np.abs(dataFFT), axis =0)/np.mean(np.abs(dataFFT_b), axis =0)

        c += 1
    f_spec = f_spec[1:, :]
    z_spec = z_spec[1:, :]
    v_spec = v_spec[1:, :]
    ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
    t = [i for i in range(int(window/2), (data.shape[2] - int(window/2)), step)]
    # t= [i for i in range(4000, 6000, 10)]
    t = np.array(t)

    ### Save the results
    np.savez(fname.replace('.avi', '_stft_window400_shift20'), ff = ff, f_spec = f_spec, z_spec= z_spec, v_spec= v_spec)
    print('The STFT data have been saved.')


    ### Plot the power spectrum
    # f_idx =np.where((ff>= 350) & (ff<=500))
    #plt.figure()

    #img = plt.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmax=3000)
    #plt.colorbar()

    #plt.figure()
    #img = plt.pcolormesh(t, ff[f_idx], z_spec[f_idx])
    #plt.colorbar()
    #plt.figure()
    #img = plt.pcolormesh(t, ff[f_idx], v_spec[f_idx], vmax=100)
    #plt.colorbar()

    plt.style.use('dark_background')
    ### Make the power spectrum video ###
    print('Create videos for the STFT data.')
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    f_idx = np.where((ff >= 0) & (ff <= 50))
    fig = plt.figure()
    ax2 = plt.subplot(122)
    ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmin=0, vmax=300)
    # ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx])

    # ln = ax2.axvline(4000, color='red')
    ln = ax2.axvline(500, color='red')
    x = 0
    ax1 = plt.subplot(121)
    im = ax1.imshow(data[:, :, 0], cmap='gray')
    ax1.axis('off')

    import pandas as pd
    fft = np.mean(f_spec, axis=1)
    df = pd.DataFrame(fft.reshape(1, len(fft)))
    fname_stft = fname.replace('.avi', '_stft.npz')
    df.insert(0, "fname", fname_stft)
    main_dir = 'Z:/HsinYi/web_vibration'

    # df.to_csv(main_dir + '/temp.csv')
    # # df.to_csv(main_dir + '/FFT_table2.csv')
    # df = pd.read_csv(main_dir + '/temp.csv', index_col=[0])
    # df2 = pd.read_csv(main_dir + '/STFT_table_prey.csv', index_col=[0])
    # frames = [df, df2]
    # result = pd.concat(frames)
    # result.to_csv(main_dir + '/STFT_table_prey.csv')

    with writer.saving(fig, fname.replace('.avi', '_stft.mp4'), 100):
        for i in range(data.shape[2]):
            x += 10
            if x > 500:
                # ln.set_xdata(3000+x)
                ln.set_xdata(x)

            if x > data.shape[2]:
                break
            im.set_data(data[:, :, x])
            writer.grab_frame()

    # from scipy import signal
    # f, t, Zxx = signal.stft(data[res[0], res[1], :], 1000, nperseg=5000)
    # plt.pcolormesh(t, f, np.mean(np.abs(Zxx), axis = 0), vmin=0, vmax =2 * np.sqrt(2))
    # plt.ylim([100, 500])
    # plt.show()

    # f, t, Sxx = signal.spectrogram(data[res[0], res[1], :], 1000, return_onesided=False)
    # plt.pcolormesh(t, f, np.mean(Sxx, axis =0))
    # plt.ylim([0, 500])

    return f_spec