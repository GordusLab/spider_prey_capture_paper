import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as manimation
import os, glob, matplotlib.pyplot as plt, scipy
from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square
import cv2

fname = 'Z:\HsinYi\web_vibration/082923/082923_spider_prey_C001H001S0001/082923_spider_prey_C001H001S0001.avi'
data = np.load(fname.replace(".avi", ".xyt") + '.npy')
filename = fname.replace(".avi", ".xyt") + '.npy.txt'


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

freq_m1 = 0
freq_m2= 50
AUC = np.load(fname.replace('.avi', '_') + str(int(freq_m1)) + '-' + str(int(freq_m2)) + 'hz_auc_timewindow40_step20.npz'
                                    )
AUC = AUC['AUC']
auc_map = np.copy(AUC)
auc_map[np.isnan(auc_map)] = 0
print('AUC max = ' + str(auc_map.max()))
if 'control' in fname:
    auc_map[np.where(auc_map > 5000)] = 5000
auc_map = auc_map / auc_map.max() * 255
auc_map = auc_map.astype(np.uint8)

images_snr = []
images_auc = []

time_window = 40
step = 20

##
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
data0 = data[:, :, 0]
writer = FFMpegWriter(fps=3, metadata=metadata)

for j in range(0, len(range(time_window, data.shape[2] - time_window, step))):
    images_auc.append(auc_map[:, :, j])

(x, y, w, h) = roi_spider_list[time_window]

#### Ploting FFT result
fig = plt.figure()
ax2 = plt.subplot()
x1 = 424
x2 = 703
y1 = 405
y2 = 594
im = ax2.imshow(data[x1:x2, y1:y2, 0], cmap='gray')
lm = ax2.imshow(images_auc[0][x1:x2, y1:y2], cmap='hot', vmin=0, vmax=40, alpha=0.6)


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
    for t in range(len(images_auc)):
        map_data = images_auc[t][x1:x2, y1:y2]
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
        # Plot the extracted signal data
        selected_data.append(line_data)

        ax2.plot(line_data)
        ax2.set_title("Extracted Signal Data along the Line")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Signal Value")
        plt.show()
        click_count = 0  # Reset click count for next line
    fig.canvas.draw()
    np.save('selected_data', np.array(selected_data))
# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)
# Show the 2D map
plt.show()


### Plot the selected radii AUC over time
from matplotlib import cm
fig3, ax3 = plt.subplots()
count=int(selected_data.shape[0]/2)+2-1
greys = cm.get_cmap('Greys', 256)
newcolors = greys(np.linspace(0, 1, int(selected_data.shape[0]/2)+2))
ax3.plot(selected_data[0], c=newcolors[count])

for i in range(1,len(selected_data),2):

    count = count-1
    ax3.plot(selected_data[i], c=newcolors[count])
    ax3.plot(selected_data[i+1], c=newcolors[count])


