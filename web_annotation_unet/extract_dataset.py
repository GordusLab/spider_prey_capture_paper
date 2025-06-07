import glob
import os
import numpy as np
import shutil

directory = 'Z:\HsinYi\web_vibration/'
datafile = glob.glob(directory+'*/*/*.npy')
save_directory = 'B:\HsinYi\web_annotation_data/'

for i in range(len(datafile)):
    fname = datafile[i]
    if os.path.exists(fname.replace('.npy', '.npy.txt')):
        # Load the full array (if stored in a .npy file)
        data= np.load(fname, mmap_mode='r')  # Memory-mapped for efficiency
        # Extract the first frame (time=0)
        first_frame = data[:, :, 0]
        textfile = fname.replace('.npy', '.npy.txt')
        destination_path =save_directory+fname.split('\\')[-1]+'txt'

        np.save(save_directory+fname.split('\\')[-1], first_frame)
        shutil.copy(textfile, destination_path)

