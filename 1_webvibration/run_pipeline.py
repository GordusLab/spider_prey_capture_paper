"""
Created on Wed Jan 05 2022

@author: Hsin-Yi
"""
from FFT_web_vibration import *
from preprocessing_avitonpy import *
from spatial_AUC_web_vibration_sliding_window import *
from STFT_web_vibration import *
from lasso_selector import SelectFromCollection
from FFT_peak import *
import pandas as pd

def run_pipeline(fname):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import interpolate as interp

    ### Preprocessing: convert the avi video into npy file and save it.
    print('Run preprocessing: converting avi to npy file')
    data = preprocessing_avitonpy(fname)

    ### Calculating the Fourier spectrum of the web vibration
    print('Run FFT analysis')
    ff, dataFFT = FFT_web_vibration(fname, npydata=data)
    
    ### Save FFT to csv table
    fft = np.mean(dataFFT, axis=0)[(ff > 0) & (ff < 100)]
    column_names = [ff[(ff > 0) & (ff < 100)]]

    if len(fft) < 873:
        arr2_interp = interp.interp1d(np.arange(fft.size), fft)
        arr2_stretch = arr2_interp(np.linspace(0, fft.size - 1, 873))
        fft = arr2_stretch
        arr2 = column_names[0]
        arr2_interp = interp.interp1d(np.arange(arr2.size), arr2)
        arr2_stretch = arr2_interp(np.linspace(0, arr2.size - 1, 873))
        column_names[0] = arr2_stretch
    df = pd.DataFrame(fft.reshape(1,len(fft)),columns=column_names)
    fname_fft = fname.replace('.avi', '_fft.npz')
    df.insert(0, "fname", fname_fft)
    main_dir = 'Z:/HsinYi/web_vibration'
    
    df.to_csv(main_dir +'/temp.csv')
    #df.to_csv(main_dir + '/FFT_table2.csv')
    df = pd.read_csv(main_dir +'/temp.csv', index_col=[0])
    df2 = pd.read_csv(main_dir +'/FFT_table2.csv', index_col=[0])
    frames = [df, df2]
    result = pd.concat(frames)
    result.to_csv(main_dir +'/FFT_table2.csv')
    
    
    #### Ploting FFT result 
    fig, ax = plt.subplots()
    ax.plot(ff[(ff > 0)], np.mean(dataFFT, axis=0)[(ff > 0)])
    selector = SelectFromCollection(ax)
    selected_region = []
    def accept(event):
        if event.key == "enter":
            selected_region.append(selector.select)
            #selector.disconnect()
            ax.set_title("Press enter to accept selected points.\nClose the window to continue the pipeline")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.\nClose the window to continue the pipeline")
    plt.show()

    print('You have selected '+str(len(selected_region))+ ' ROIs in the FFT spectrum.')


    ### Run the AUC analysis for the web vibration
    for i in range(len(selected_region)):
        print('Process the selected ROI ' + str(i))
        selected_x = selected_region[i][:,0]
        freq_m1 = np.rint(np.min(selected_x))
        freq_m2 = np.rint(np.max(selected_x))
        FFT_peak(freq_m1, freq_m2 , fname)
        auc = spatial_AUC_web_vibration_sliding_window(fname, freq_m1, freq_m2, npydata=data)

    ### Run the STFT analysis for the web vibration
    print('Run STFT analysis')
    f_spec = STFT_web_vibration(fname, npydata=data)

    print('Finish running the pipleline!')