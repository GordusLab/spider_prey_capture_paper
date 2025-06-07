import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as manimation
import os, glob, matplotlib.pyplot as plt, scipy
from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square
import math
import cv2



fname ='Z:\HsinYi\web_vibration/082923/082923_spider_prey_C001H001S0001/082923_spider_prey_C001H001S0001.avi'
data = np.load(fname.replace(".avi", ".xyt") + '.npy')
filename = fname.replace(".avi", ".xyt") + '.npy.txt'

def create_circle_mask(size, center, radius):
    y, x = np.ogrid[:size[0], :size[1]]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = distance <= radius
    return mask

### Get web by annotaation ###
if os.path.exists(filename):
    annotations = loadAnnotations(filename)
    lines = annotations[0][3]
    points = annotations[0][1]

    webmask = np.full((data.shape[0], data.shape[1]), False, dtype=np.bool)
    for line in lines:
        rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
        webmask[rr, cc] = True

    for point in points:
        webmask[point[0], point[1]] = True
    webmask_origin = np.copy(webmask)
    webmask = skimage.morphology.dilation(webmask, square(3))
elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300_modified.npy")):
    webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300_modified.npy"))
    webmask_origin = np.copy(webmask)
elif os.path.exists(fname.replace(".avi", "_get_unet_acc_n300.npy")):
    webmask = np.load(fname.replace(".avi", "_get_unet_acc_n300.npy"))
    webmask_origin = np.copy(webmask)
else:
    raise NotImplementedError("No annotation file found.")

roi_spider_list = np.load(fname.replace('.avi', '_spider_roi_coordinates.npz'))
roi_spider_list = roi_spider_list['roi_spider_list']
res_spider_peripheral = []
signal_map = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
data2 = np.copy(data)
kernel = np.ones((3, 3), np.uint8)
for z in range(data.shape[2] ):
    webmask_temp = np.copy(webmask)
    (x, y, w, h) = roi_spider_list[z]

    ##### Spider peripheral

    circle_mask_peripheral = create_circle_mask((data.shape[1], data.shape[0]), (int((2 * x + w) / 2),
                                                                                 int((2 * y + h) / 2)),
                                                int((w + h) / 2))
    circle_mask_peripheral = np.transpose(circle_mask_peripheral)
    circle_mask = create_circle_mask((data.shape[1], data.shape[0]), (int((2 * x + w) / 2),
                                                                      int((2 * y + h) / 2)), int((w + h) / 4))
    circle_mask = np.transpose(circle_mask)
    circle_mask = ~circle_mask
    combined_mask = circle_mask & circle_mask_peripheral
    combined_mask = combined_mask & webmask_temp
    res_spider_peripheral_temp = np.where(combined_mask == True)
    res_spider_peripheral.append(res_spider_peripheral_temp)
    res = res_spider_peripheral_temp
    res_origin = np.copy(res)
    erosion = cv2.erode(data[:, :, z], kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    data[:, :, z] = data[:, :, z] - dilation
    signal_map[res[0], res[1], z] = np.abs(data[res[0], res[1], z])

np.savez(fname.replace('.avi', '_signal_map_data.npz'), signal_map)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
data = np.copy(data2)
data0 = data[:, :, 0]
writer = FFMpegWriter(fps=50, metadata=metadata)


(x, y, w, h) = roi_spider_list[0]

#### Ploting FFT result
fig = plt.figure()
ax2 = plt.subplot()
x1 = 414
x2 = 843
y1 = 260
y2 = 570
im = ax2.imshow(data2[x1:x2, y1:y2, 0], cmap='gray')
lm = ax2.imshow(signal_map[x1:x2, y1:y2, 0], cmap='hot', vmin=0, vmax=30, alpha=0.4)


plt.axis('off')
# lm = ax2.imshow(data[:, :, 0], cmap='gray', alpha =0.2)
plt.tight_layout()
# List to store the line data
line_data = []
# Function to extract signal data along the line
def extract_line_data(start, end):
    """Extracts the signal data along a line from start to end."""
    # Calculate the number of points to extract based on the distance between start and end


    num_points = int(np.linalg.norm(np.array(end) - np.array(start)))  # Euclidean distance
    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    # Extract signal values from the 2D map based on the line coordinates
    signal_values_t = []
    for t in range(signal_map.shape[2]):
        map_data = signal_map[x1:x2, y1:y2, t]
        signal_values = []
        for x, y in zip(x_vals, y_vals):
            xi, yi = int(np.round(x)), int(np.round(y))  # Round to the nearest integer indices
            if 0 <= xi < map_data.shape[0] and 0 <= yi < map_data.shape[1]:
                signal_values.append(map_data[ yi, xi])


        signal_values_t.append(np.mean(signal_values))
    return signal_values_t
# Line object to draw the line
from matplotlib import lines
line = lines.Line2D([0, 0], [0, 0], color='blue', lw=2, ls='--')
ax2.add_line(line)
# Function to update the line and extract data when mouse clicks
click_count = 0
start_point = None
fig2, ax2 = plt.subplots()
selected_data = []
radii_data=[]
def on_click(event):
    """Called when the user clicks to draw the line."""
    global click_count, start_point
    if click_count == 0:  # First click: define start point
        start_point = (event.xdata, event.ydata)
        line.set_xdata([start_point[0], start_point[0]])  # Set x data for line start
        line.set_ydata([start_point[1], start_point[1]])  # Set y data for line start
        click_count += 1
    elif click_count == 1:  # Second click: define end point
        end_point = (event.xdata, event.ydata)
        line.set_xdata([start_point[0], end_point[0]])  # Set x data for line end
        line.set_ydata([start_point[1], end_point[1]])  # Set y data for line end
        # Extract signal data along the line
        line_data = extract_line_data(start_point, end_point)
        radii_data.append([start_point, end_point])
        # Plot the extracted signal data
        selected_data.append(line_data)

        ax2.plot(line_data)
        ax2.set_title("Extracted Signal Data along the Line")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Signal Value")
        plt.show()
        click_count = 0  # Reset click count for next line
    fig.canvas.draw()
    #np.savez(fname.replace('.avi','_signal_map_selected_data.npz'), np.array(selected_data))
# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)
# Show the 2D map
plt.show()


import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manimation
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip

## Specify ffmpeg on HsinYi's macbook pro
# plt.rcParams['animation.ffmpeg_path'] = '/Users/hsinyihung/opt/anaconda3/bin/ffmpeg'
##
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
data0 = data2[:, :, 0]
writer = FFMpegWriter(fps=50, metadata=metadata)



x1 = 414
x2 = 843
y1 = 260
y2 = 570
fig = plt.figure()
ax2 = plt.subplot(121)
z=0

im = ax2.imshow(data2[x1:x2, y1:y2, z], cmap='gray')
lm = ax2.imshow(signal_map[x1:x2, y1:y2, z], cmap='hot', vmin=0, vmax=30, alpha=0.7)
ax2.axis('off')
# lm = ax2.imshow(data[:, :, 0], cmap='gray', alpha =0.2)
ax3 = plt.subplot(122)
ln = ax3.axvline(z, color='red')
ax3.plot(selected_data[0,:], label ='Radii 0')
ax3.plot(selected_data[1,:], label ='Radii 1')
ax3.plot(selected_data[2,:], label ='Radii 2')
ax3.plot(selected_data[15,:], label ='Radii 15')
ax3.plot(selected_data[3,:], label ='Radii 3')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Average pixel intensity along radii')
plt.tight_layout()

fig_manager = plt.get_current_fig_manager()
fig_manager.full_screen_toggle()


with writer.saving(fig, fname.replace(".avi", "_signal_map.mp4") , 200):
    for z in range(data2.shape[2]):

        lm.set_data(signal_map[x1:x2, y1:y2, z])

        im.set_data(data2[x1:x2, y1:y2, z])
        ln.set_xdata(z)

        writer.grab_frame()
