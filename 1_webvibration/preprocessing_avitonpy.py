# -*- coding: utf-8 -*-
"""
Created on Wed Jan 05 2022

@author: Hsin-Yi

Preprocessing: convert the avi file to npy
"""

def preprocessing_avitonpy(fname=None):
    import os
    import imageio
    from moviepy.editor import VideoFileClip
    import numpy as np

    ### Convert the video to python data
    if fname is None:
        raise NotImplementedError("Invalid filename for preprocessing. Please select an avi video.")
    elif not fname.lower().endswith('.avi'):
        raise Exception('Unsupported input file. Only *.avi files allowed.')
    else:
        if os.path.exists(fname.replace(".avi", ".xyt") + '.npy'):
            print("The data have already been preprocessed. Load data.")
            data = np.load(fname.replace(".avi", ".xyt") + '.npy')

        elif os.path.exists(fname.replace(".avi", "_nohand.xyt") + '.npy'):
            # print("The data have already been preprocessed. Load data.")
            data = np.load(fname.replace(".avi", "_nohand.xyt") + '.npy')

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
            

        return data
