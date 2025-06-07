import glob
import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
directory = 'Z:\HsinYi\web_vibration/'
datafile = glob.glob(directory+'042225/*/*.xyt.npy')
save_directory = 'B:\HsinYi\web_annotation_data/'
model = tf.keras.models.load_model('get_unet_acc_n300_modified.keras')

## Show the model architecture
model.summary()


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask



for i in tqdm(range(len(datafile))):
    fname = datafile[i]
    if os.path.exists(fname.replace('.npy', '.npy.txt')):
        continue
    else:
        # Load the full array (if stored in a .npy file)
        # Load the full array (if stored in a .npy file)
        data= np.load(fname, mmap_mode='r')  # Memory-mapped for efficiency
        # Extract the first frame (time=0)
        first_frame = data[:, :, 0]

        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(first_frame, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        img = first_frame - dilation
        circlemask = create_circular_mask(img.shape[0], img.shape[1],
                                          center=(int(img.shape[1] / 2), int(img.shape[0] / 2)), radius=425)
        img[circlemask == False] = 0
        img = img.reshape(-1, 1280, 1024, 1)
        predictions = model.predict(img)
        mask = (predictions> 0.5).astype(int)
        mask = mask[0,:,:,0]
        np.save(fname.replace('.xyt.npy', '_get_unet_acc_n300_modified.npy'), mask)
        print('Save '+ fname.replace('.xyt.npy', '_get_unet_acc_n300_modified.npy'))