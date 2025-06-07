
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manimation
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip

## Specify ffmpeg on HsinYi's macbook pro
# plt.rcParams['animation.ffmpeg_path'] = '/Users/hsinyihung/opt/anaconda3/bin/ffmpeg'
##


fname ='Z:\HsinYi\web_vibration/082923/082923_spider_prey_C001H001S0001/082923_spider_prey_C001H001S0001.avi'
signal_map = np.load(fname.replace('.avi','_signal_map.npz'))
signal_map = signal_map['signal_map']
signal_data = np.load(fname.replace('.avi','_signal_map_radii_data.npz'))
signal_data = signal_data['arr_0']
data2 = np.load(fname.replace(".avi", ".xyt") + '.npy')


FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
data0 = data2[:, :, 0]
writer = FFMpegWriter(fps=50, metadata=metadata)


plt.style.use('dark_background')
x1 = 414
x2 = 843
y1 = 260
y2 = 570
fig = plt.figure()
ax2 = plt.subplot(121)
z=0

im = ax2.imshow(data2[x1:x2, y1:y2, z], cmap='gray')
lm = ax2.imshow(signal_map[x1:x2, y1:y2, z], cmap='hot', vmin=0, vmax=30, alpha=0.7)
ax2.axis('off')
# lm = ax2.imshow(data[:, :, 0], cmap='gray', alpha =0.2)
ax3 = plt.subplot(122)
ln = ax3.axvline(z, color='red')
ax3.plot(signal_data[0,:], label ='Radii 0')
ax3.plot(signal_data[1,:], label ='Radii 1')
ax3.plot(signal_data[4,:], label ='Radii 4')
ax3.plot(signal_data[15,:], label ='Radii 15')
ax3.plot(signal_data[2,:], label ='Radii 2')
ax3.set_xlabel('Time (ms)')

ax3.set_ylabel('Average pixel intensity along radii')
plt.tight_layout()

fig_manager = plt.get_current_fig_manager()
fig_manager.full_screen_toggle()


with writer.saving(fig, fname.replace(".avi", "_signal_map.mp4") , 200):
    for z in range(data2.shape[2]):

        lm.set_data(signal_map[x1:x2, y1:y2, z])

        im.set_data(data2[x1:x2, y1:y2, z])
        ln.set_xdata(z)

        writer.grab_frame()
