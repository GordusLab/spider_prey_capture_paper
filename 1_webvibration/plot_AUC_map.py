"""
# -*- coding: utf-8 -*-
Created on Mon Feb 21 09:19:53 2022

@author: Hsin-Yi
"""
import matplotlib.pyplot as plt
import skimage.draw, numpy as np

from scipy import stats


fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/0606_spider001_spider_prey_C001H001S0001/0606_spider001_spider_prey_C001H001S0001.avi'
freq_m1 = 4
freq_m2 = 17

auc = np.load(fname.replace('.avi', '_')+str(freq_m1)+'-'+str(freq_m2)+'hz_auc.npz')
auc_data  = auc['AUC']


AUC = auc_data
auc_map = np.copy(AUC)
auc_map[np.isnan(auc_map)] = 0
print('AUC max = ' + str(auc_map.max()))
if 'control' in fname:
    auc_map[np.where(auc_map>5000)]=5000
auc_map = auc_map / auc_map.max() * 255
auc_map = auc_map.astype(np.uint8)

plt.figure()
plt.imshow(auc_map[:,:,26], alpha=1, cmap='hot')
plt.colorbar()
plt.savefig(fname.replace('.avi', '_auc.png'), dpi = 3000)

data = np.load(fname.replace(".avi", ".xyt") + '.npy')
plt.figure()
plt.imshow(data[:, :, 0], cmap='gray')
plt.savefig(fname.replace(".avi", "_web.png"), dpi = 3000)

stft= np.load(fname.replace(".avi", "_stft.npz"))
ff = stft['ff']
f_spec = stft['f_spec']
t = [i for i in range(500, (data.shape[2] - 500), 10)]
f_idx = np.where((ff >= 0) & (ff <= 50))
fig = plt.figure()
ax2 = plt.subplot(122)
ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmin=0, vmax=3000)
plt.savefig(fname.replace(".avi", "_stft.png"), dpi = 3000)


