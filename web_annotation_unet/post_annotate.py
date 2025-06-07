import matplotlib
matplotlib.use('TkAgg')
import  numpy as np, matplotlib.pyplot as plt, scipy
import skimage.draw
from skimage.morphology import square
from loadAnnotations import *

fname = 'Z:/HsinYi/web_vibration/022725/022725_spider_prey_v3_C001H001S0001/022725_spider_prey_v3_C001H001S0001.avi'
mask = np.load(fname.replace('.avi', '_get_unet_acc_n300_modified.npy'))
data = np.load(fname.replace('.avi', '.xyt.npy'), mmap_mode='r')
annotations = loadAnnotations(fname.replace(".avi", "_get_roipreymask.npy.txt"))
lines = annotations[0][3]
points = annotations[0][1]
webmask = np.full((data.shape[0], data.shape[1]), False, dtype=np.bool_)

for line in lines:
    rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])

    webmask[rr, cc] = True

for point in points:
    webmask[point[0], point[1]] = True

webmask = skimage.morphology.dilation(webmask, square(3))


mask_modified = np.copy(mask)
idx = np.where(webmask == True)
for i in range(len(idx[0])):
    if mask_modified[idx[0][i], idx[1][i]] == 0:
        mask_modified[idx[0][i], idx[1][i]] = 1
import matplotlib

fig = plt.figure()
plt.imshow(data[:,:,0])
plt.imshow(mask_modified, alpha=0.5)
np.save(fname.replace('.avi','_get_unet_acc_n300_modified.npy'),
    mask_modified)
np.save(fname.replace('.avi','_0.xyt.npy'),
    data[:,:,0])