# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 2022

@author: Hsin-Yi

Calculating the Area under the FFT spectrum between two frequencies across the web.
"""
from get_video import *
from loadAnnotations import *
from preprocessing_avitonpy import *
import cv2
import numpy as np
fname = 'Z:\HsinYi\web_vibration/101723/1017233 Spider Prey Pt1-10172023162754-0000.avi'
video_path = 'Z:\HsinYi\web_vibration/101723/1017233 Spider Prey Pt1-10172023162754-0000.mp4'
def select_roi(video_path):
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()


    roi = cv2.selectROI("Select ROI: spider", frame, fromCenter=False, showCrosshair=True)

    cv2.destroyAllWindows()
    return roi

def roi_frame(video_path):

    tracker_type = 'KCF'  # You can change the tracker type here if needed

    # Create a VideoCapture object and select ROI 1
    roi1 = select_roi(video_path)

    # Create another VideoCapture object to read the video again for ROI 2
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    roi2 = cv2.selectROI("Select ROI: fly", frame, fromCenter=False, showCrosshair=True)

    cv2.destroyAllWindows()

    # Initialize two separate trackers with the chosen tracker type
    tracker1 = cv2.TrackerKCF_create()
    tracker2 = cv2.TrackerKCF_create()

    # Read the first frame of the video and initialize the trackers with the ROIs
    _, frame = video.read()
    tracker1.init(frame, roi1)
    tracker2.init(frame, roi2)

    # List to store the ROIs for all frames
    # (x, y, w, h) = [int(coord) for coord in roi1]
    # roi1 = (y, x, h, w)
    # (x, y, w, h) = [int(coord) for coord in roi2]
    # roi2 = (y, x, h, w)

    roi1_list = [roi1]
    roi2_list = [roi2]
    roi1_list.append(roi1)
    roi2_list.append(roi2)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' key to exit
            break
        elif key == ord('s'):
            roi1 = cv2.selectROI("Select ROI: spider", frame, fromCenter=False, showCrosshair=True)

            tracker1 = cv2.TrackerKCF_create()
            tracker1.init(frame, roi1)
        elif key == ord('f'):
            roi2 = cv2.selectROI("Select ROI: fly", frame, fromCenter=False, showCrosshair=True)

            tracker2 = cv2.TrackerKCF_create()
            tracker2.init(frame, roi2)

        _, frame = video.read()
        if not _:
            break

        # Update the trackers and get the new bounding box coordinates
        success1, new_roi1 = tracker1.update(frame)
        success2, new_roi2 = tracker2.update(frame)
        # (x, y, w, h) = [int(coord) for coord in new_roi1]
        # new_roi1 = (y, x, h, w)
        # (x, y, w, h) = [int(coord) for coord in new_roi2]
        # new_roi2 = (y, x, h, w)
        if success1:
            # If tracking is successful for ROI 1, draw the bounding box around it
            (x, y, w, h) = [int(coord) for coord in new_roi1]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4), (0, 255, 0), 2)
            roi1_list.append(new_roi1)
            roi1_save = new_roi1
        else:

            (x, y, w, h) = [int(coord) for coord in roi1_save]
            #roi1 = cv2.selectROI("Select ROI: spider", frame, fromCenter=False, showCrosshair=True)
            cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4), (0, 255, 0), 2)
            roi1_list.append(roi1_save)
            #roi1_list.append(new_roi1)
            #roi1_save = new_roi1
            #roi1_list.append(roi1)
            #tracker1 = cv2.TrackerKCF_create()
            #tracker1.init(frame, roi1)
            #cv2.destroyAllWindows()


        if success2:
            # If tracking is successful for ROI 2, draw the bounding box around it
            (x, y, w, h) = [int(coord) for coord in new_roi2]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4), (0, 255, 0), 2)
            roi2_list.append(new_roi2)
            roi2_save = new_roi2
            print('frame : ' + str(video.get(cv2.CAP_PROP_POS_FRAMES)))
        else:
            (x, y, w, h) = [int(coord) for coord in roi2_save]
            cv2.circle(frame, (int((2 * x + w) / 2), int((2 * y + h) / 2)), int((w + h) / 4), (0, 255, 0), 2)
            roi2_list.append(roi2_save)
            # roi2_list.append(roi2_save)
            # roi2 = cv2.selectROI("Select ROI: fly", frame, fromCenter=False, showCrosshair=True)
            # roi2_list.append(roi2)
            # tracker2 = cv2.TrackerKCF_create()
            # tracker2.init(frame, roi2)
            # cv2.destroyAllWindows()

        cv2.imshow("Object Tracking", frame)

        #if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        #    break
        #elif int(video.get(cv2.CAP_PROP_POS_FRAMES))==nframes:
        #    break
        if int(video.get(cv2.CAP_PROP_POS_FRAMES))==nframes:
            break

    video.release()
    cv2.destroyAllWindows()

    # Save the ROIs to separate files (e.g., text files)
    np.savez(video_path.replace('.mp4',"_roi_coordinates_side.npz"),roi_spider_list = roi1_list,  roi_fly_list = roi2_list, nframes = nframes)

    #return roi2_list
    return roi1_list, roi2_list, nframes

def create_circle_mask(size, center, radius):
    y, x = np.ogrid[:size[0], :size[1]]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = distance <= radius
    return mask

def FFT_partial_web(res_fly, res_spider, res_fly_peripheral, res_spider_peripheral, data):
    import numpy as np, matplotlib.pyplot as plt, scipy
    from scipy import stats
    from scipy import fft
    ###### The FFT analysis
    #### Fly
    temp_fly = res_fly

    # Ensure all arrays inside the tuples have the same length (optional if already done)
    min_length = min(len(arr) for tup in temp_fly for arr in tup)
    temp_fly = [tuple(arr[:min_length] for arr in tup) for tup in temp_fly]
    fly = np.zeros((min_length, len(temp_fly)))

    #### Fly p
    temp_fly_p = res_fly_peripheral

    # Ensure all arrays inside the tuples have the same length (optional if already done)
    min_length = min(len(arr) for tup in temp_fly_p for arr in tup)
    temp_fly_p = [tuple(arr[:min_length] for arr in tup) for tup in temp_fly_p]
    fly_p = np.zeros((min_length, len(temp_fly_p)))

    #### Spider
    temp_spider = res_spider

    # Ensure all arrays inside the tuples have the same length (optional if already done)
    min_length = min(len(arr) for tup in temp_spider for arr in tup)
    temp_spider = [tuple(arr[:min_length] for arr in tup) for tup in temp_spider]
    spider = np.zeros((min_length, len(temp_spider)))

    #### Spider p
    temp_spider_p = res_spider_peripheral

    # Ensure all arrays inside the tuples have the same length (optional if already done)
    min_length = min(len(arr) for tup in temp_spider_p for arr in tup)
    temp_spider_p = [tuple(arr[:min_length] for arr in tup) for tup in temp_spider_p]
    spider_p = np.zeros((min_length, len(temp_spider_p)))

    for j in range(len(temp_fly)):
        fly[:, j] = data[ temp_fly[j][0], temp_fly[j][1], j ]
        fly_p[:, j] = data[temp_fly_p[j][0], temp_fly_p[j][1], j ]
        spider[:, j] = data[temp_spider[j][0], temp_spider[j][1], j ]
        spider_p[:, j] = data[temp_spider_p[j][0], temp_spider_p[j][1], j]
    # Convert the list of tuples with arrays to a NumPy array
    # temp2 = np.array(temp2)
    dataFFT_fly = np.mean(np.abs(scipy.fft.fft(fly)), axis=0)
    dataFFT_fly_p = np.mean(np.abs(scipy.fft.fft(fly_p)), axis=0)
    dataFFT_spider = np.mean(np.abs(scipy.fft.fft(spider)), axis=0)
    dataFFT_spider_p = np.mean(np.abs(scipy.fft.fft(spider_p)), axis=0)

    ### Save the results
    ff = np.fft.fftfreq(dataFFT_fly.shape[0], 0.01)
    np.savez(fname.replace('.avi', '_roi_fft_side'), ff=ff, dataFFT_fly=dataFFT_fly, dataFFT_fly_p =dataFFT_fly_p , dataFFT_spider=dataFFT_spider, dataFFT_spider_p=dataFFT_spider_p)
    print('The partial FFT data have been saved.')



    plt.figure()
    plt.plot(ff[ff > 0], dataFFT_fly[ff > 0])
    plt.savefig(fname.replace('.avi', '_roi_fly_fft_side.png'), dpi=300)
    plt.figure()
    plt.plot(ff[ff > 0], dataFFT_fly_p[ff > 0])
    plt.savefig(fname.replace('.avi', '_roi_fly_p_fft_side.png'), dpi=300)
    plt.figure()
    plt.plot(ff[ff > 0], dataFFT_spider[ff > 0])
    plt.savefig(fname.replace('.avi', '_roi_spider_fft_side.png'), dpi=300)
    plt.figure()
    plt.plot(ff[ff > 0], dataFFT_spider_p[ff > 0])
    plt.savefig(fname.replace('.avi', '_roi_spider_p_fft_side.png'), dpi=300)
def STFT_partial_web(res_fly, res_spider, res_fly_peripheral, res_spider_peripheral, data, window, step):
    import numpy as np, matplotlib.pyplot as plt, scipy
    from scipy import stats
    from scipy import fft

    f_spec_fly = np.zeros((window, len(range(window, data.shape[2], step))))
    f_spec_fly_p = np.zeros((window, len(range(window, data.shape[2], step))))
    f_spec_spider = np.zeros((window, len(range(window, data.shape[2], step))))
    f_spec_spider_p = np.zeros((window, len(range(window, data.shape[2], step))))
    # f_spec = np.zeros((1000, len(range(4000, 6000, 10))))

    c = 0
    for t in range(int(window/2), (data.shape[2] - int(window/2)), step):
        #### Fly
        temp_fly = res_fly[(t - int(window/2)):(t + int(window/2))]

        # Ensure all arrays inside the tuples have the same length (optional if already done)
        min_length = min(len(arr) for tup in temp_fly for arr in tup)
        temp_fly = [tuple(arr[:min_length] for arr in tup) for tup in temp_fly]
        fly = np.zeros((min_length, len(temp_fly)))

        #### Fly p
        temp_fly_p = res_fly_peripheral[(t - int(window/2)):(t + int(window/2))]

        # Ensure all arrays inside the tuples have the same length (optional if already done)
        min_length = min(len(arr) for tup in temp_fly_p for arr in tup)
        temp_fly_p = [tuple(arr[:min_length] for arr in tup) for tup in temp_fly_p]
        fly_p = np.zeros((min_length, len(temp_fly_p)))

        #### Spider
        temp_spider = res_spider[(t - int(window/2)):(t + int(window/2))]

        # Ensure all arrays inside the tuples have the same length (optional if already done)
        min_length = min(len(arr) for tup in temp_spider for arr in tup)
        temp_spider = [tuple(arr[:min_length] for arr in tup) for tup in temp_spider]
        spider = np.zeros((min_length, len(temp_spider)))

        #### Spider p
        temp_spider_p = res_spider_peripheral[(t - int(window/2)):(t + int(window/2))]

        # Ensure all arrays inside the tuples have the same length (optional if already done)
        min_length = min(len(arr) for tup in temp_spider_p for arr in tup)
        temp_spider_p = [tuple(arr[:min_length] for arr in tup) for tup in temp_spider_p]
        spider_p = np.zeros((min_length, len(temp_spider_p)))


        for j in range(len(temp_fly)):
            fly[:,j] = data[ temp_fly[j][0], temp_fly[j][1], j+t-int(window/2)]
            fly_p[:, j] = data[ temp_fly_p[j][0], temp_fly_p[j][1], j+t-int(window/2)]
            spider[:, j] = data[ temp_spider[j][0], temp_spider[j][1], j+t-int(window/2)]
            spider_p[:, j] = data[ temp_spider_p[j][0], temp_spider_p[j][1], j+t-int(window/2)]
        # Convert the list of tuples with arrays to a NumPy array
        #temp2 = np.array(temp2)
        dataFFT = np.abs(scipy.fft.fft(fly))

        f_spec_fly[:, c] = np.mean(dataFFT, axis=0)
        dataFFT = np.abs(scipy.fft.fft(fly_p))
        f_spec_fly_p[:, c] = np.mean(dataFFT, axis=0)


        dataFFT = np.abs(scipy.fft.fft(spider))
        f_spec_spider[:, c] = np.mean(dataFFT, axis=0)
        dataFFT = np.abs(scipy.fft.fft(spider_p))
        f_spec_spider_p[:, c] = np.mean(dataFFT, axis=0)


        # dataFFT_b = np.abs(scipy.fft(baseline_data[res[0], res[1], (t-1000):t]))
        # f_spec[:,c] = np.mean(np.abs(dataFFT), axis =0)/np.mean(np.abs(dataFFT_b), axis =0)

        c += 1
    f_spec_fly = f_spec_fly[1:, :]
    f_spec_fly_p = f_spec_fly_p[1:, :]
    f_spec_spider = f_spec_spider[1:, :]
    f_spec_spider_p = f_spec_spider_p[1:, :]

    #ff = np.fft.fftfreq(dataFFT.shape[1], 1/(len(temp_fly)))
    ff = np.fft.fftfreq(dataFFT.shape[1], 0.01)
    t = [i for i in range(int(window/2), (data.shape[2] - int(window/2)), step)]
    # t= [i for i in range(4000, 6000, 10)]
    t = np.array(t)

    ### Save the results
    np.savez(fname.replace('.avi', '_roi_stft_side'), ff=ff, f_spec_fly=f_spec_fly, f_spec_spider=f_spec_spider, f_spec_fly_p=f_spec_fly_p, f_spec_spider_p=f_spec_spider_p)
    print('The partial STFT data have been saved.')

    ### Plot the power spectrum

    f_idx = np.where((ff >= 0) & (ff <= 50))
    fig = plt.figure()
    ax2 = plt.subplot()
    ax2.pcolormesh(t, ff[f_idx], f_spec_fly[f_idx], vmin=0, vmax=200)
    plt.savefig(fname.replace('avi', '_roi_fly_stft_side.png'), dpi =300)
    fig = plt.figure()
    ax2 = plt.subplot()
    ax2.pcolormesh(t, ff[f_idx], f_spec_fly_p[f_idx], vmin=0, vmax=200)
    plt.savefig(fname.replace('avi', '_roi_fly_p_stft_side.png'), dpi=300)
    fig = plt.figure()
    ax2 = plt.subplot()
    ax2.pcolormesh(t, ff[f_idx], f_spec_spider[f_idx], vmin=0, vmax=200)
    plt.savefig(fname.replace('avi', '_roi_spider_stft_side.png'), dpi=300)
    fig = plt.figure()
    ax2 = plt.subplot()
    ax2.pcolormesh(t, ff[f_idx], f_spec_spider_p[f_idx], vmin=0, vmax=200)
    plt.savefig(fname.replace('avi', '_roi_spider_p_stft_side.png'), dpi=300)


def Partialweb_vibration_analysis(fname, video_path, npydata=None):
    import os, numpy as np, matplotlib.pyplot as plt, scipy
    from moviepy.editor import VideoFileClip
    from scipy import stats
    from scipy import fft
    import skimage.draw
    from skimage.morphology import square


    ### Check if the STFT analyis has been saved ###
    if os.path.exists(fname.replace('.avi', '_roi_stft_side.npz')):
        spec = np.load(fname.replace('.avi', '_roi_stft_side.npz'))
        f_spec_fly = spec['f_spec_fly']
        f_spec_spider = spec['f_spec_spider']
        f_spec_fly_p = spec['f_spec_fly_p']
        f_spec_spider_p = spec['f_spec_spider_p']
        ff = spec['ff']
        print('The STFT data already exist.')
        #return f_spec_fly, f_spec_spider, f_spec_fly_p, f_spec_spider_p, ff

    ### Read the npy data ###

    if fname is None or not fname.lower().endswith('.avi'):
        raise NotImplementedError("Invalid filename for preprocessing. Please select an avi video.")

    #
    # filename = fname.replace(".avi", ".xyt") + '.npy.txt'
    # ### Get web by annotaation ###
    # if os.path.exists(filename):
    #     annotations = loadAnnotations(filename)
    # else:
    #     raise NotImplementedError("No annotation file found.")
    #
    # lines = annotations[0][3]
    # points = annotations[0][1]

    if os.path.exists(video_path.replace('.mp4',"_roi_coordinates_side.npz")):
        roi = np.load(video_path.replace('.mp4',"_roi_coordinates_side.npz"))
        roi = np.load(video_path.replace('.mp4',"_roi_coordinates_side.npz"))
        roi_spider_list = roi['roi_spider_list']
        roi_fly_list = roi['roi_fly_list']
        nframes = roi['nframes']
    else:
        roi_spider_list, roi_fly_list, nframes = roi_frame(video_path)

    data = get_video(video_path, nframes, 0, nframes)
    data = data[:, :, :, 0]
    data = np.swapaxes(data, 0, 2)

    #np.save(fname.replace('.avi', '.xyt.npy'), data)
    #
    # webmask = np.full((data.shape[0], data.shape[1]), False, dtype=np.bool_)
    #
    # for line in lines:
    #     rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
    #
    #     webmask[rr, cc] = True
    #
    #
    # for point in points:
    #
    #     webmask[point[0], point[1]] = True
    #
    # webmask = skimage.morphology.dilation(webmask, square(3))
    #

    res_spider=[]
    res_spider_peripheral=[]
    res_fly=[]
    res_fly_peripheral=[]

    for z in range(data.shape[2]-1):
        # webmask_temp = np.copy(webmask)
        (x,y,w,h) = roi_spider_list[z]
        ##### Spider center
        circle_mask = create_circle_mask((data.shape[1], data.shape[0]), (int((2 * x + w) / 2),
                                         int((2 * y + h) / 2)), int((w + h) / 4))
        circle_mask = np.transpose(circle_mask)
        # combined_mask = circle_mask & webmask_temp
        combined_mask = circle_mask

        res_spider_temp = np.where(combined_mask == True)
        res_spider.append(res_spider_temp)

        ##### Spider peripheral

        circle_mask_peripheral = create_circle_mask((data.shape[1], data.shape[0]), (int((2 * x + w) / 2),
                                                                          int((2 * y + h) / 2)), int((w + h) / 2))
        circle_mask_peripheral = np.transpose(circle_mask_peripheral)
        circle_mask = ~circle_mask
        combined_mask = circle_mask & circle_mask_peripheral
        # combined_mask = combined_mask & webmask_temp

        res_spider_peripheral_temp = np.where(combined_mask == True)
        res_spider_peripheral.append(res_spider_peripheral_temp)

        #webmask_temp = np.copy(webmask)
        (x, y, w, h) = roi_fly_list[z]
        ##### Fly center
        circle_mask = create_circle_mask((data.shape[1], data.shape[0]), (int((2 * x + w) / 2),
                                         int((2 * y + h) / 2)), int((w + h) / 4))
        circle_mask = np.transpose(circle_mask)
        # combined_mask = circle_mask & webmask_temp
        combined_mask = circle_mask

        res_fly_temp = np.where(combined_mask == True)
        res_fly.append(res_fly_temp)

        ##### Spider peripheral

        circle_mask_peripheral = create_circle_mask((data.shape[1], data.shape[0]), (int((2 * x + w) / 2),
                                                                          int((2 * y + h) / 2)), int((w + h) / 2))
        circle_mask_peripheral = np.transpose(circle_mask_peripheral)
        circle_mask = ~circle_mask
        combined_mask = circle_mask & circle_mask_peripheral
        # combined_mask = combined_mask & webmask_temp

        res_fly_peripheral_temp = np.where(combined_mask == True)
        res_fly_peripheral.append(res_fly_peripheral_temp)





    ### Extract the web index
    # bottom right portion of the web
    # data = data[200:600, 400:800, :]

    ### Substract the spider/flies/stabalimentum
    # kernel = np.ones((5, 5), np.uint8)
    # for i in range(0, data.shape[2], 500):
    #    erosion = cv2.erode(data[:, :, i:(i + 500)], kernel, iterations=1)
    #    dilation = cv2.dilate(erosion, kernel, iterations=1)
    #    data[:, :, i:(i + 500)] = data[:, :, i:(i + 500)] - dilation
    # data[:, :, i:(i+500) ] = dilation

    ### Extract the web index
    # threshold = 20
    # Use maximum projection as a thresold
    # maxprojection = np.amax(data, axis =2)

    FFT_partial_web(res_fly, res_spider, res_fly_peripheral, res_spider_peripheral, data)
    STFT_partial_web(res_fly, res_spider, res_fly_peripheral, res_spider_peripheral, data, window=40, step=2)




    ##### Plot movies
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as manimation

    # ### Make the power spectrum video ###
    # print('Create videos for the STFT data.')
    # FFMpegWriter = manimation.writers['ffmpeg']
    # metadata = dict(title='Movie Test', artist='Matplotlib',
    #                 comment='Movie support!')
    # writer = FFMpegWriter(fps=15, metadata=metadata)
    #
    # fig = plt.figure()
    # ax2 = plt.subplot(122)
    # ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmin=0, vmax=3000)
    # # ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx])
    #
    # # ln = ax2.axvline(4000, color='red')
    # ln = ax2.axvline(200, color='red')
    # x = 0
    # ax1 = plt.subplot(121)
    # im = ax1.imshow(data[:, :, 0], cmap='gray')
    #
    # with writer.saving(fig, fname.replace('.avi', '_stft.mp4'), 100):
    #     for i in range(data.shape[2]):
    #         x += 10
    #         if x > 500:
    #             # ln.set_xdata(3000+x)
    #             ln.set_xdata(x)
    #
    #         if x > data.shape[2]:
    #             break
    #         im.set_data(data[:, :, x])
    #         writer.grab_frame()




Partialweb_vibration_analysis(fname, video_path)