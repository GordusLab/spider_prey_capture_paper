import glob
import numpy as np
import os, matplotlib.pyplot as plt, scipy
files = glob.glob('B:\HsinYi\DeepLabCut_Anthony\8videos_1400frames_relabled/videos/2023dlc_model/aligned/011822 Spider Prey*_croprot*.npy')
cut = 569

for i in range(len(files)):
    data = np.load(files[i])
    nframe = data.shape[1]
    data = data[:,cut:nframe]

    np.save(files[i], data)
