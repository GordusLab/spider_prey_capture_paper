# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 2022

@author: Hsin-Yi

Calculating the Area under the FFT spectrum between two frequencies across the web.
"""



import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
fname = 'Z:\HsinYi\web_vibration/012122/012122_spider_prey/012122_spider_prey.avi'
video_path = 'Z:\HsinYi\web_vibration/012122/012122_spider_prey/012122 Spider Prey.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20


data = np.load(fname.replace( '.avi','_stft_window400_shift20.npz'))

all_web_ff = data['ff']
all_web_stft = data['f_spec']

f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))

coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']

data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(200, (data.shape[2] - 200), 20)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)





plt.style.use('dark_background')
fig = plt.figure()
px = 1/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(1920*px, 969*px))
axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
im=axs['Left'].imshow(data[:,:,0], cmap='gray')

(y, x, h, w) = roi_fly_list[0]
drawObject3 = Circle((int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 2),color = 'red', fill=False)
axs['Left'].add_patch(drawObject3)
drawObject4 = Circle((int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4),color = 'red', fill=False)
axs['Left'].add_patch(drawObject4)
axs['Left'].axis('off')

axs['TopRight'].set_title('Web STFT', fontsize=10)
cf = axs['TopRight'].pcolormesh(t, all_web_ff[f_idx], all_web_stft[f_idx], vmin=0, vmax=1000)
#cf = axs['TopRight'].pcolormesh(roi_t, roi_ff[roi_f_idx], roi_f_spec_spider[roi_f_idx], vmin=0, vmax=2500)
# cbar2= plt.colorbar(cf)
# cbar2.ax.tick_params(labelsize=6)
fig.colorbar(cf, ax=axs['TopRight'])
#axs['TopRight'].set_xticklabels([])
axs['TopRight'].set_yticklabels([0,10,20,30,40,50],fontsize=10)
ln = axs['TopRight'].axvline(200, color='red')

axs['BottomRight'].set_title('Normalized fly STFT', fontsize=10)
cd=axs['BottomRight'].pcolormesh(roi_t, roi_ff[roi_f_idx], normalized_fly[roi_f_idx], vmin=0, vmax=10)
fig.colorbar(cd, ax=axs['BottomRight'])
#cbar=plt.colorbar(cd)
#cbar.ax.tick_params(labelsize=6)
axs['BottomRight'].set_yticklabels([0,10,20,30,40,50],fontsize=10)
#axs['BottomRight'].set_xticklabels([0,0,5000],fontsize=6)
ln2 = axs['BottomRight'].axvline(200, color='red')
plt.tight_layout()



import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
writer = FFMpegWriter(fps=250, metadata=metadata)

with writer.saving(fig, fname.replace('.avi', '_roi_stft.mp4'), 200):
    #for i in range(6000):
    for i in range(data.shape[2]-1):

        #if x > 500:
            # ln.set_xdata(3000+x)
        if i>200:
            ln.set_xdata(i)
            ln2.set_xdata(i)

        if x > data.shape[2]:
            break
        im.set_data(data[:, :, i])
        (y, x, h, w) = roi_spider_list[i]
        # drawObject1.center = (int((2 * x + w) / 2), int((2 * y + h) / 2))
        # drawObject1.radius =  int((w + h) / 2)
        # drawObject2.center =(int((2 * x + w) / 2), int((2 * y + h) / 2))
        # drawObject2.radius = int((w + h) / 4)

        (y, x, h, w) = roi_fly_list[i]

        drawObject3.center = (int((2 * x + w) / 2), int((2 * y + h) / 2))
        drawObject3.radius = int((w + h) / 2)

        drawObject4.center=(int((2 * x + w) / 2), int((2 * y + h) / 2))
        drawObject4.radius = int((w + h) / 4)

        writer.grab_frame()

