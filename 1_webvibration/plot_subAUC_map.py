import os, glob
import imageio
import numpy as np
import matplotlib.pyplot as plt


auc = np.load('Z:\HsinYi\web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey_4-9hz_auc.npz')
auc = auc['AUC']
data = np.load('Z:\HsinYi\web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey.xyt.npy')
t = np.arange(1000, data.shape[2], 100)
auc_control = np.load('Z:\HsinYi\web_vibration/070121/0701_spider003_web2_control_air/0701_spider003_web2_control_air_4-9hz_auc.npz')
auc_control = auc_control['AUC']
test = np.subtract(auc/auc.max(), auc_control/auc_control.max())
test2 = test/test.max()

test2[np.where(test2 == 0) ]= np.nan
plt.figure()
plt.imshow(test2[:,:,46], alpha=1, cmap='hot', vmin=0, vmax=0.1, zorder=1)
plt.colorbar(ticks=[0,0.1])
plt.imshow( data[:, :, 5600], alpha=1, cmap='gray', zorder=-1)
plt.xticks([])
plt.yticks([])

plt.savefig('Z:\HsinYi\web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey_4-9hz_subauc.png', dpi = 300)
