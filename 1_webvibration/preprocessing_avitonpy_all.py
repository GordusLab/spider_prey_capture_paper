import os, glob
import imageio
from moviepy.editor import VideoFileClip
import numpy as np

### Load files
main_dir = 'Z:\HsinYi\web_vibration/051425/*'
#files = glob.glob(main_dir+'/*/*.avi')
files = glob.glob(main_dir+'/*.avi')
for fname in files:
    
    ### Convert the video to python data
    if fname is None:
        raise NotImplementedError("Invalid filename for preprocessing. Please select an avi video.")
    elif not fname.lower().endswith('.avi'):
        raise Exception('Unsupported input file. Only *.avi files allowed.')
    else:
        if os.path.exists(fname.replace(".avi", ".xyt") + '.npy'):
            # print("The data have already been preprocessed. Load data.")
            # data = np.load(fname.replace(".avi", ".xyt") + '.npy')
            continue
        elif os.path.exists(fname.replace(".avi", "_nohand.xyt") + '.npy'):
            # print("The data have already been preprocessed. Load data.")
            # data = np.load(fname.replace(".avi", ".xyt") + '.npy')
            continue
        else:
            video = VideoFileClip(fname)
            r = imageio.get_reader(fname)

            data = np.zeros((video.size[0], video.size[1], video.reader.nframes), dtype=np.uint8)
            idx = 0
            for frame in r.iter_data():
                if video.size[0] == video.size[1]:
                    data[:, :, idx] = np.mean(frame, axis=2)
                else:
                    data[:, :, idx] = np.mean(np.transpose(frame, (1, 0, 2)), axis=2)
                idx += 1
            np.save(fname.replace(".avi", ".xyt") + '.npy', data)
