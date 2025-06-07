# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 2022

@author: Hsin-Yi

Calculating the Area under the FFT spectrum between two frequencies across the web.
"""

from get_video import *
from loadAnnotations import *
from preprocessing_avitonpy import *
import cv2
import numpy as np
from cv2 import VideoWriter_fourcc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


fname = 'Z:\HsinYi\web_vibration/021424/021424 Spider Prey Pt1-02142024162226-0000.avi'
video_path = 'Z:\HsinYi\web_vibration/021424/021424 Spider Prey Pt1-02142024162226-0000.mp4'
stft =np.load(fname.replace( '.avi','_roi_stft_side.npz'))
fft =np.load(fname.replace( '.avi','_roi_fft_side.npz'))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates_side.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
nframes = roi_spider_list.shape[0]
data = get_video(video_path, nframes, 0, nframes)
data= data[:,:,:,0]
data = np.swapaxes(data, 0, 2)
#data = np.load(fname.replace('.avi', '.xyt.npy'))


ff_stft = stft['ff']
f_spec_spider_p =stft['f_spec_spider_p']
f_spec_fly_p =stft['f_spec_fly_p']


#ff_fft = fft['ff']
#dataFFT = fft['dataFFT']




plt.style.use('dark_background')
fig = plt.figure()
axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
im=axs['Left'].imshow(data[:,:,0], cmap='gray')
(y, x, h, w) = roi_spider_list[0]
drawObject1 = Circle((int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 2),color = 'green', fill=False)
axs['Left'].add_patch(drawObject1)
drawObject2 = Circle((int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4),color = 'green', fill=False)
axs['Left'].add_patch(drawObject2)
(y, x, h, w) = roi_fly_list[0]
drawObject3 = Circle((int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 2),color = 'red', fill=False)
axs['Left'].add_patch(drawObject3)
drawObject4 = Circle((int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4),color = 'red', fill=False)
axs['Left'].add_patch(drawObject4)
axs['Left'].axis('off')
axs['TopRight'].set_title('Fly peripheral STFT', fontsize=10)
t = [i for i in range(int(40 / 2), (data.shape[2] - int(40 / 2)), 2)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
f_idx = np.where((ff_stft >= 0) & (ff_stft <= 50))
cf = axs['TopRight'].pcolormesh(t, ff_stft[f_idx], f_spec_fly_p[f_idx], vmin=0, vmax=200)
cbar2= plt.colorbar(cf)
cbar2.ax.tick_params(labelsize=6)
axs['TopRight'].set_xticklabels([])
axs['TopRight'].set_yticklabels([0,0,10,20,30,40,50],fontsize=6)
ln = axs['TopRight'].axvline(20, color='red')
axs['BottomRight'].set_title('Spider peripheral STFT', fontsize=10)
t = [i for i in range(int(40 / 2), (data.shape[2] - int(40 / 2)), 2)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
f_idx = np.where((ff_stft >= 0) & (ff_stft <= 50))
cd=axs['BottomRight'].pcolormesh(t, ff_stft[f_idx], f_spec_spider_p[f_idx], vmin=0, vmax=4000)
cbar=plt.colorbar(cd)
cbar.ax.tick_params(labelsize=6)
axs['BottomRight'].set_yticklabels([0,0,10,20,30,40,50],fontsize=6)
#axs['BottomRight'].set_xticklabels([0,0,5000],fontsize=6)
ln2 = axs['BottomRight'].axvline(20, color='green')



import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
writer = FFMpegWriter(fps=250, metadata=metadata)

with writer.saving(fig, fname.replace('.avi', '_roi_center_stft_side.mp4'), 200):
    #for i in range(6000):
    for i in range(data.shape[2]-1):

        #if x > 500:
            # ln.set_xdata(3000+x)
        if i>20:
            ln.set_xdata(i)
            ln2.set_xdata(i)

        if x > data.shape[2]:
            break
        im.set_data(data[:, :, i])
        (y, x, h, w) = roi_spider_list[i]
        drawObject1.center = (int((2 * x + w) / 2), int((2 * y + h) / 2))
        drawObject1.radius =  int((w + h) / 2)
        drawObject2.center =(int((2 * x + w) / 2), int((2 * y + h) / 2))
        drawObject2.radius = int((w + h) / 4)

        (y, x, h, w) = roi_fly_list[i]

        drawObject3.center = (int((2 * x + w) / 2), int((2 * y + h) / 2))
        drawObject3.radius = int((w + h) / 2)

        drawObject4.center=(int((2 * x + w) / 2), int((2 * y + h) / 2))
        drawObject4.radius = int((w + h) / 4)

        writer.grab_frame()

# import cv2
# def main():
#     # reading the input
#     cap = cv2.VideoCapture(video_path)
#     fourcc =cv2.VideoWriter_fourcc(*'mp4v')
#     output = cv2.VideoWriter(
#         "output.mp4", fourcc, 500, (1024, 1280))
#     i=0
#     while (True):
#         ret, frame = cap.read()
#         if (ret):
#             # adding rectangle on each frame
#             (x, y, w, h) = roi_spider_list[i]
#             cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 2), (0, 255, 0), 2)
#
#             cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4), (0, 255, 0), 2)
#             (x, y, w, h) = roi_fly_list[i]
#             cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4), (0, 0, 255), 2)
#             cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 2), (0, 0, 255), 2)
#             i=i+1
#             # writing the new frame in output
#             output.write(frame)
#             #cv2.imshow("output", frame)
#             #if cv2.waitKey(1):
#             #    break
#             print(i)
#     cv2.destroyAllWindows()
#     output.release()
#     cap.release()
# if __name__ == "__main__":
#     main()
#
#
