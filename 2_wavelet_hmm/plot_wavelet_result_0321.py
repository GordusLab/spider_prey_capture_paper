
import numpy as np
import pandas as pd
from get_video import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import matplotlib.pyplot as plt
velocity_window = 5 # 50ms
wavelet_window = 50  # 500ms


videodirectory = 'B:\HsinYi\DeepLabCut_Anthony\8videos_1400frames_relabled/videos/aligned/'
videoname = '032122 Spider Prey-03212022154120-0000-1'
stftname = '032122l_Spider_prey_stft'
videofile = ['B:\HsinYi\DeepLabCut_Anthony\8videos_1400frames_relabled/videos/2023dlc_model/032122 Spider Prey-03212022154120-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000_labeled.mp4']

## load wavelet data
wavelet = np.load(videodirectory+ 'wavelet/'+videoname+ '_croprotaligned01234_wavelet.npz')
wavelet_x = wavelet['arr_0']
wavelet_y = wavelet['arr_1']
ff = wavelet['arr_2']

### Ignore wrapping part
wavelet_x = wavelet_x[0:3792,:]
wavelet_y = wavelet_y[0:3792,:]



### Make the power spectrum video ###

import matplotlib.animation as manimation
print('Create videos for the wavelet cluster visualization')
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=50, metadata=metadata)

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation



data_joints = np.load(videodirectory+ 'croprot/'+videoname+ '_croprotaligned.npy')
clip_time = data_joints.shape[0]

umap_data = np.load('B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/2023dlc_model/aligned/results_summary/umap_hmmpredict_results.npz')
umap = umap_data['umap']
labeled_beh_extend_num = umap_data['manual_label']
hmm_label = umap_data['hmm_label']

buf = get_video(videofile, clip_time, 0, clip_time)

### starting from the frame that drop the fly (remove hand movement at the begingning of recordings)
start_frame = 567

plt.style.use('dark_background')
px = 1/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(1920*px, 969*px))
axs = fig.subplot_mosaic([['Left', 'TopMiddle', 'TopRight'],['Left', 'TopMiddle', 'BottomRight']],
                          gridspec_kw={'width_ratios':[1, 1,1]})

im = axs['Left'].imshow(np.fliplr(np.rot90(buf[start_frame,:,:,:])), cmap = 'gray',aspect="auto")
axs['Left'].axes.get_xaxis().set_visible(False)
axs['Left'].axes.get_yaxis().set_visible(False)
axs['Left'].set_axis_off()

from matplotlib.pyplot import get_cmap
rainbow = get_cmap('rainbow', 256)
newcolors = rainbow(np.linspace(0, 1, 6))

sc = axs['BottomRight'].scatter(np.reshape(umap,(98444,5))[np.where(labeled_beh_extend_num==0)[0],0],
                  np.reshape(umap,(98444,5))[np.where(labeled_beh_extend_num==0)[0],1],  s=0.5, c=newcolors[1],zorder=1)
sc = axs['BottomRight'].scatter(np.reshape(umap,(98444,5))[np.where(labeled_beh_extend_num==1)[0],0],
                  np.reshape(umap,(98444,5))[np.where(labeled_beh_extend_num==1)[0],1],  s=0.5, c='yellow',zorder=1)
sc = axs['BottomRight'].scatter(np.reshape(umap,(98444,5))[np.where(labeled_beh_extend_num==2)[0],0],
                  np.reshape(umap,(98444,5))[np.where(labeled_beh_extend_num==2)[0],1],  s=0.5, c=newcolors[4],zorder=1)

# labels=['Static','Crouching', 'High frequency']
# axs['Right'].legend(labels, loc="right", fontsize=8)


im2 = axs['BottomRight'].scatter(0,0,marker = "*", s=200, c = 'red',  edgecolor='red',zorder=1)
axs['BottomRight'].set_title('Manual label')
axs['BottomRight'].axes.get_xaxis().set_visible(False)
axs['BottomRight'].axes.get_yaxis().set_visible(False)
axs['BottomRight'].set_axis_off()




sct = axs['TopRight'].scatter(np.reshape(umap,(98444,5))[np.where(hmm_label==0)[0],0],
                  np.reshape(umap,(98444,5))[np.where(hmm_label==0)[0],1],  s=0.5, c=newcolors[1],zorder=1)
sct = axs['TopRight'].scatter(np.reshape(umap,(98444,5))[np.where(hmm_label==1)[0],0],
                  np.reshape(umap,(98444,5))[np.where(hmm_label==1)[0],1],  s=0.5, c='yellow',zorder=1)
sct = axs['TopRight'].scatter(np.reshape(umap,(98444,5))[np.where(hmm_label==2)[0],0],
                  np.reshape(umap,(98444,5))[np.where(hmm_label==2)[0],1],  s=0.5, c=newcolors[4],zorder=1)

# labels=['Static','Crouching', 'High frequency']
# axs['Right'].legend(labels, loc="right", fontsize=8)


im2t = axs['TopRight'].scatter(0,0,marker = "*", s=200, c = 'red',  edgecolor='red',zorder=1)
axs['TopRight'].set_title('HMM prediction')
axs['TopRight'].axes.get_xaxis().set_visible(False)
axs['TopRight'].axes.get_yaxis().set_visible(False)
axs['TopRight'].set_axis_off()


##### Plot wavelet
## only use half of wavelet matrix to exclude noise
indices = np.concatenate([np.arange(start, start + 25) for start in range(25, 250, 50)])
wavelet_x_half = wavelet_x[:,indices]
lm2 = axs['TopMiddle'].pcolormesh(np.array(list(np.arange(0, wavelet_x_half.shape[0]))),
                     np.array(list(np.arange(0, 25*5))), wavelet_x_half.T, vmax=0.5)
# ax2.xaxis.set_ticks([])
#axs['TopMiddle'].set_title("Wavelet transform: x coordinates")
# plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
ln2 = axs['TopMiddle'].axvline(0, color='red')
# axs['TopMiddle'].axes.get_xaxis().set_visible(False)
axs['TopMiddle'].set_xticks([1000, 2000, 3000])
axs['TopMiddle'].set_xticklabels( [10, 20, 30])
axs['TopMiddle'].set_xlabel( 'Time (s)')
axs['TopMiddle'].set_ylabel( 'Frequency (HZ)')
axs['TopMiddle'].set_yticks([0,10, 25,35, 50, 60, 75, 85, 100, 110, 125])
axs['TopMiddle'].set_yticklabels( [2.38, 'Co-Tr', 50,'Fe-Pa',  50,'Ti-Me', 50,'Me-Ta',  50, 'Ta-Cl',50])
# axs['TopMiddle'].axes.get_yaxis().set_visible(False)



import glob
videodirectory = 'B:\HsinYi\DeepLabCut_Anthony\8videos_1400frames_relabled/videos/2023dlc_model/aligned/'
videoname = '*'
filenames=glob.glob(videodirectory+ 'wavelet/'+videoname+ '_croprotaligned01234_nonormalized_wavelet.npz')
filenames=[ x for x in filenames if "crouched" not in x ]
filenames=[ x for x in filenames if "Crouched" not in x ]
csv_filenames = glob.glob('B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/2023dlc_model/aligned/merging_static/' + '*' + '_wavelet_timestep.csv')
clip_time_2 =[]
for i in range(len(filenames)):
    wavelet = np.load(filenames[i])
    wavelet_x_temp = wavelet['arr_0']
    # f = wavelet['arr_2']
    clip_time_temp = wavelet_x_temp.shape[0]

    test_wavelet = wavelet_x_temp.reshape(int(clip_time_temp), 5, 50)
    ## half wavelet x
    ## try cutting first and end 1s
    # test_wavelet = test_wavelet[100:int(clip_time-100),:,25:50]
    test_wavelet = test_wavelet[:, :, 25:50]
    test_wavelet = test_wavelet.reshape(int(clip_time_temp), 25 * 5)
    beh = np.load(
        filenames[i].split('/wavelet\\')[0] + '/wavelet/' + filenames[i].split('/wavelet\\')[1].split('01234')[
            0] + '01234_nonormalized_wavelet_manuallabels.npy'
    )
    test_wavelet = test_wavelet[0:len(beh), :]
    clip_time_2.append(len(test_wavelet))

    if i == 0:
        data_sample = test_wavelet
    else:
        data_sample = np.append(data_sample, test_wavelet, axis=0)
clip_time_2 = np.array(clip_time_2)
clip_time_2 = np.cumsum(clip_time_2)


savevideoname = '032122 Spider Prey-03212022154120-0000-1_umap.mp4'
with writer.saving(fig, savevideoname, 200):

    for i in range(len(wavelet_x)):

        freq = i
        im.set_data(np.fliplr(np.rot90(buf[start_frame+freq,:,:,:])))

        freq_idx = freq



        #if freq > 50:
        #    ln.set_xdata(freq*10)
        time_idx = clip_time_2[4]+freq
        ln2.set_xdata(freq)

        im2.set_offsets((np.reshape(umap,(98444,5))[time_idx,0], np.reshape(umap,(98444,5))[time_idx, 1]))
        im2t.set_offsets((np.reshape(umap, (98444, 5))[time_idx, 0], np.reshape(umap, (98444, 5))[time_idx, 1]))

        writer.grab_frame()



