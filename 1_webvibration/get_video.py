

def get_video(files_mp4, clip_time, time_start_for_file, time_end_for_file):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.animation as manimation
    import matplotlib.pyplot as plt
    import imageio
    from moviepy.editor import VideoFileClip

    vid_name = files_mp4
    vid = imageio.get_reader(vid_name, 'ffmpeg')
    import cv2

    from matplotlib.animation import FuncAnimation

    cap = cv2.VideoCapture(vid_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((int(clip_time), frameHeight, frameWidth, 3), np.dtype('uint8'))
    cap = cv2.VideoCapture(vid_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_start_for_file))
    fc = 0
    ret = True
    while (fc < int(clip_time) and ret):
        ret, buf[fc] = cap.read()
        buf[fc] = cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
        fc += 1
    cap.release()

    return buf