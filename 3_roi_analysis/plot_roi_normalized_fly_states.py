
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import scipy.stats as st

fname = 'Z:\HsinYi\web_vibration/093024/093024_spider_prey_pt1_t4_C001H001S0001/093024_spider_prey_pt1_t4_C001H001S0001_nohand.avi'
video_path = 'Z:\HsinYi\web_vibration/093024//093024_spider_prey_pt1_t4_C001H001S0001/093024 Spider Prey Pt1 T4 C001h001s0001_nohand.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(500, (data.shape[2] - 500), 10)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
#plt.style.use('dark_background')
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 0:91], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 90:123], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 125:196], axis=1), label='Turning')
#
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 199:219], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 236:263], axis=1), label='Shaking 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 265:274], axis=1), label='Crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 324:361], axis=1), label='Shaking 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx,362:383], axis=1), label='Crouch 4')
plt.legend()
plt.show()
Crouch=[]
Static_after_crouch=[]
Static=[]
Static.append( np.mean(normalized_fly[roi_f_idx, 0:91], axis=1))
Shake=[]
Crouch.append( np.mean(normalized_fly[roi_f_idx, 90:123], axis=1))
Crouch.append( np.mean(normalized_fly[roi_f_idx, 199:219], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 236:263], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 265:274], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 324:361], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx,362:383], axis=1))


fname = 'Z:\HsinYi\web_vibration/061921/0619_spider002_spider_prey_top_C001H001S0001/0619_spider002_spider_prey_top_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/061921/0619_spider002_spider_prey_top_C001H001S0001/0619_spider002_spider_prey_top_C001H001S0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(500, (data.shape[2] - 500), 10)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 0:29], axis=1)/np.mean(normalized_fly[0, 0:29]), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 28:109], axis=1)/np.mean(normalized_fly[0, 28:109]), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 108:129], axis=1)/np.mean(normalized_fly[0, 108:129]), label='Static after crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 128:206], axis=1)/np.mean(normalized_fly[0, 128:206]), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 205:217], axis=1)/np.mean(normalized_fly[0, 205:217]), label='Static after crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 216:313], axis=1)/np.mean(normalized_fly[0, 216:313]), label='Crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 314:359], axis=1)/np.mean(normalized_fly[0, 314:359]), label='Static after crouch 3')
plt.legend()
plt.show()

Static_after_shake=[]
Shake2=[]
Static.append(np.mean(normalized_fly[roi_f_idx, 0:29], axis=1))
Shake2.append(np.mean(normalized_fly[roi_f_idx, 28:104], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 104:109], axis=1))
Static_after_shake.append(np.mean(normalized_fly[roi_f_idx, 108:129], axis=1))
Shake2.append(np.mean(normalized_fly[roi_f_idx, 128:201], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 201:206], axis=1))
Static_after_shake.append(np.mean(normalized_fly[roi_f_idx, 205:217], axis=1))
Shake2.append(np.mean(normalized_fly[roi_f_idx, 216:308], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 308:313], axis=1))
Static_after_shake.append(np.mean(normalized_fly[roi_f_idx, 314:359], axis=1))




fname = 'Z:\HsinYi\web_vibration/081823/081823_spider_prey_C001H001S0001/081823_spider_prey_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/081823/081823_spider_prey_C001H001S0001/081823 Spider Prey C001h001s0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(500, (data.shape[2] - 500), 10)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 89:212], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 211:258], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 257:271], axis=1), label='Static after crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 270:306], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 305:322], axis=1), label='Static after crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 321:337], axis=1), label='Crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 336:355], axis=1), label='Static after crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 354:384], axis=1), label='Crouch 4')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 383:388], axis=1), label='Static after crouch 4')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 387:400], axis=1), label='Crouch 5')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 399:417], axis=1), label='Static after crouch 5')
plt.legend()
plt.show()

Static.append(np.mean(normalized_fly[roi_f_idx, 89:212], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 211:258], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 257:271], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 270:306], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 305:322], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 321:337], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 336:355], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 354:384], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 383:388], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 387:400], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 399:417], axis=1))



fname = 'Z:\HsinYi\web_vibration/081623/081623_spider_prey_C001H001S0001/081623_spider_prey_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/081623/081623_spider_prey_C001H001S0001/081623 Spider Prey C001h001s0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(500, (data.shape[2] - 500), 10)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 77:241], axis=1)/np.mean(normalized_fly[0, 77:241]), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 302:323], axis=1)/np.mean(normalized_fly[0, 302:323]), label='Shaking')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 330:384], axis=1)/np.mean(normalized_fly[0, 330:384]), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 383:409], axis=1)/np.mean(normalized_fly[0, 383:409]), label='Static after crouch 1')

# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 77:241], axis=1)/np.mean(normalized_fly[0, 77:241]), label='Static')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 332:340], axis=1)/np.mean(normalized_fly[0, 332:340]), label='Crouch 1')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 339:342], axis=1)/np.mean(normalized_fly[0, 339:342]), label='Static after crouch 1')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 341:349], axis=1)/np.mean(normalized_fly[0, 341:349]), label='Crouch 2')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 348:352], axis=1)/np.mean(normalized_fly[0, 348:352]), label='Static after crouch 2')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 351:362], axis=1)/np.mean(normalized_fly[0, 351:362]), label='Crouch 3')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 369:377], axis=1)/np.mean(normalized_fly[0, 369:377]), label='Crouch 4')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 376:391], axis=1)/np.mean(normalized_fly[0, 376:391]), label='Static after crouch 4')
plt.legend()
plt.show()
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 77:241], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 302:323], axis=1), label='Shaking')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 330:384], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 383:409], axis=1), label='Static after crouch 1')
plt.legend()
plt.show()
Static.append(np.mean(normalized_fly[roi_f_idx, 77:241], axis=1))
Shake.append( np.mean(normalized_fly[roi_f_idx, 302:323], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 330:384], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 383:409], axis=1))
from loadAnnotations import *
from preprocessing_avitonpy import *
import cv2
import numpy as np
from cv2 import VideoWriter_fourcc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
fname = 'Z:\HsinYi\web_vibration/012122/012122_spider_prey/012122_spider_prey.avi'
video_path = 'Z:\HsinYi\web_vibration/012122/012122_spider_prey/012122 Spider Prey.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 0:85], axis=1)/np.mean(normalized_fly[0, 0:85]), label='Turning')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 113:154], axis=1)/np.mean(normalized_fly[0,113:154]), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 172:236], axis=1)/np.mean(normalized_fly[0, 172:236]), label='Static after crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 240:284], axis=1)/np.mean(normalized_fly[0, 240:284]), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 335:355], axis=1)/np.mean(normalized_fly[0, 335:355]), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 356:375], axis=1)/np.mean(normalized_fly[0, 356:375]), label='Crouch 3')
plt.legend()
plt.show()
Crouch.append(np.mean(normalized_fly[roi_f_idx, 113:154], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 172:236], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 240:284], axis=1))

# Crouch.append(np.mean(normalized_fly[roi_f_idx, 356:375], axis=1))



fname = 'Z:\HsinYi\web_vibration/093024/093024_spider_prey_pt1_t4_C001H001S0001/093024_spider_prey_pt1_t4_C001H001S0001_nohand.avi'
video_path = 'Z:\HsinYi\web_vibration/093024/093024_spider_prey_pt1_t4_C001H001S0001/093024 Spider Prey Pt1 T4 C001h001s0001_nohand.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 0:19], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 19:39], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 40:259], axis=1), label='Static after crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 258:331], axis=1), label='Shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 330:346], axis=1), label='Static after shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 345:363], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 363:404], axis=1), label='Static after crouch 2')
plt.legend()
plt.show()
Static.append(np.mean(normalized_fly[roi_f_idx, 0:19], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 19:39], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 40:259], axis=1))
Shake2.append(np.mean(normalized_fly[roi_f_idx, 258:326], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 326:331], axis=1))
Static_after_shake.append(np.mean(normalized_fly[roi_f_idx, 330:346], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 345:363], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 363:404], axis=1))


fname = 'Z:\HsinYi\web_vibration/082924/082924_spider_prey_pt1_C001H001S0001/082924_spider_prey_pt1_C001H001S0001_nohand.avi'
video_path = 'Z:\HsinYi\web_vibration/082924/082924_spider_prey_pt1_C001H001S0001/082924 Spider Prey Pt1 C001h001s0001_nohand.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx], normalized_fly[roi_f_idx, 0], label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 1:80], axis=1), label='Shake 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 80:87], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 87:117], axis=1), label='Walk')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 117:128], axis=1), label='Crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 128:195], axis=1), label='Shake 2')
plt.legend()
plt.show()
Crouch.append(normalized_fly[roi_f_idx, 0])
Shake.append(np.mean(normalized_fly[roi_f_idx, 1:80], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 80:87], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 117:128], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 128:195], axis=1))

fname = 'Z:\HsinYi\web_vibration/081924/081924_spider_prey_pt1_C001H001S0001/081924_spider_prey_pt1_C001H001S0001_nohand.avi'
video_path = 'Z:\HsinYi\web_vibration/081924/081924_spider_prey_pt1_C001H001S0001/081924 Spider Prey Pt1 C001h001s0001_nohand.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx],  np.mean(normalized_fly[roi_f_idx, 0:158], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 158:165], axis=1), label='Courch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 165:181], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 181:195], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 195:206], axis=1), label='Turning')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 206:235], axis=1), label='Shake 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 235:242], axis=1), label='Crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 242:252], axis=1), label='Shake 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 253:259], axis=1), label='Crouch 4')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 259:275], axis=1), label='Shake 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 275:280], axis=1), label='Crouch 5')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 281:324], axis=1), label='Shake 4')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 325:343], axis=1), label='Static after shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 343:351], axis=1), label='Crouch 6')
plt.legend()
plt.show()
Static.append(np.mean(normalized_fly[roi_f_idx, 0:158], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 158:165], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 165:181], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 181:195], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 206:235], axis=1))
Crouch.append( np.mean(normalized_fly[roi_f_idx, 235:242], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 242:252], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 253:259], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 259:275], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 275:280], axis=1))
Shake2.append(np.mean(normalized_fly[roi_f_idx, 281:319], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 319:324], axis=1))
Static_after_shake.append(np.mean(normalized_fly[roi_f_idx, 325:343], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 343:351], axis=1))


fname = 'Z:\HsinYi\web_vibration/083023/083023_spider_prey_C001H001S0001/083023_spider_prey_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/083023/083023_spider_prey_C001H001S0001/083023 Spider Prey C001h001s0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx],  np.mean(normalized_fly[roi_f_idx, 0:170], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 170:201], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 201:250], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 250:350], axis=1), label='Walking')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 380:400], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 401:417], axis=1), label='Static after crouch')
plt.legend()
plt.show()

Static.append( np.mean(normalized_fly[roi_f_idx, 0:170], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 170:201], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 201:250], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 380:400], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 401:417], axis=1))




fname = 'Z:\HsinYi\web_vibration/082923/082923_spider_prey_C001H001S0001/082923_spider_prey_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/082923/082923_spider_prey_C001H001S0001/082923 Spider Prey C001h001s0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx],  np.mean(normalized_fly[roi_f_idx, 0:24], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 31:43], axis=1), label='Turning')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 43:73], axis=1), label='Shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 73:106], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 106:130], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 131:161], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 161:221], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 221:251], axis=1), label='Crouch 3')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 251:261], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 261:281], axis=1), label='Crouch 4')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 281:338], axis=1), label='Shake ')
plt.legend()
plt.show()
Static.append(np.mean(normalized_fly[roi_f_idx, 0:30], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 37:61], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 61:92], axis=1))
Static_after_crouch.append( np.mean(normalized_fly[roi_f_idx, 92:131], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 131:161], axis=1))
Static_after_crouch.append( np.mean(normalized_fly[roi_f_idx, 161:221], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 221:251], axis=1))
Static_after_crouch.append( np.mean(normalized_fly[roi_f_idx, 251:261], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 261:281], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 281:338], axis=1))


fname = 'Z:\HsinYi\web_vibration/082823/082823_spider_prey_C001H001S0001/082823_spider_prey_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/082823/082823_spider_prey_C001H001S0001/082823 Spider Prey C001h001s0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx],  np.mean(normalized_fly[roi_f_idx, 0:179], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 180:206], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 206:228], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 228:236], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 236:245], axis=1), label='Turning')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 245:298], axis=1), label='Shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 298:305], axis=1), label='Crouch 3')
plt.legend()
plt.show()
Static.append(np.mean(normalized_fly[roi_f_idx, 0:179], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 180:206], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 206:228], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 228:236], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx, 245:298], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 298:305], axis=1))


fname = 'Z:\HsinYi\web_vibration/070324/07032024_spider_piezo_prey_pt1_C001H001S0001/07032024_spider_piezo_prey_pt1_C001H001S0001_nohand.avi'
video_path = 'Z:\HsinYi\web_vibration/070324/07032024_spider_piezo_prey_pt1_C001H001S0001/07032024 Spider Piezo Prey Pt1 C001h001s0001_nohand.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(250, (data.shape[2] - 250), 125)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]

plt.plot(roi_ff[roi_f_idx],  np.mean(normalized_fly[roi_f_idx, 0:85], axis=1), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 85:101], axis=1), label='Turning')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 101:124], axis=1), label='Shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 125:137], axis=1), label='Static after shake')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 136:168], axis=1), label='Crouch 1')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 168:198], axis=1), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 198:228], axis=1), label='Crouch 2')
plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx,228:274], axis=1), label='Shake')
plt.legend()
plt.show()
Static.append( np.mean(normalized_fly[roi_f_idx, 0:85], axis=1))
Shake2.append(np.mean(normalized_fly[roi_f_idx, 101:119], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 119:124], axis=1))
Static_after_shake.append(np.mean(normalized_fly[roi_f_idx, 125:137], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 136:168], axis=1))
Static_after_crouch.append(np.mean(normalized_fly[roi_f_idx, 168:198], axis=1))
Crouch.append(np.mean(normalized_fly[roi_f_idx, 198:228], axis=1))
Shake.append(np.mean(normalized_fly[roi_f_idx,228:274], axis=1))

# fname = 'Z:\HsinYi\web_vibration/071024/07102024_spider_prey_pt1_C001H001S0001/07102024_spider_prey_pt1_C001H001S0001_nohand.avi'
# video_path = 'Z:\HsinYi\web_vibration/071024/07102024_spider_prey_pt1_C001H001S0001/07102024 Spider Prey Pt1 C001h001s0001_nohand.mp4'
# data =np.load(fname.replace( '.avi','_roi_stft.npz'))
# roi_ff = data['ff']
# roi_f_spec_fly = data['f_spec_fly']
# roi_f_spec_fly_p = data['f_spec_fly_p']
# roi_f_spec_spider = data['f_spec_spider']
# normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
# window=400
# step=20
# data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
# all_web_ff = data['ff']
# all_web_stft = data['f_spec']
# f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
# coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
# roi_spider_list=coordinates['roi_spider_list']
# roi_fly_list=coordinates['roi_fly_list']
# data = np.load(fname.replace('.avi', '.xyt.npy'))
# roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# # t= [i for i in range(4000, 6000, 10)]
# roi_t = np.array(roi_t)
# roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
# t = [i for i in range(250, (data.shape[2] - 250), 125)]
# # t= [i for i in range(4000, 6000, 10)]
# t = np.array(t)
# fig = plt.figure()
# roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
# roi_f_idx = roi_f_idx[0]
#
# plt.plot(roi_ff[roi_f_idx],  np.mean(normalized_fly[roi_f_idx, 0:31], axis=1), label='Static')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 31:67], axis=1), label='Shake')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 67:98], axis=1), label='Static after shake')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 98:114], axis=1), label='Crouch 1')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 114:149], axis=1), label='Static after crouch')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 149:176], axis=1), label='Crouch 2')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 176:208], axis=1), label='Static after crouch')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 208:231], axis=1), label='Crouch 3')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 231:262], axis=1), label='Static after crouch')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 262:278], axis=1), label='Crouch 4')
# plt.legend()
# plt.show()


## 0820 has resonance artifact
# fname = 'Z:\HsinYi\web_vibration/082024/082024_spider_prey_pt1_C001H001S0001/082024_spider_prey_pt1_C001H001S0001.avi'
# video_path = 'Z:\HsinYi\web_vibration/082024/082024_spider_prey_pt1_C001H001S0001/082024 Spider Prey Pt1 C001h001s0001.mp4'
# data =np.load(fname.replace( '.avi','_roi_stft.npz'))
# roi_ff = data['ff']
# roi_f_spec_fly = data['f_spec_fly']
# roi_f_spec_fly_p = data['f_spec_fly_p']
# roi_f_spec_spider = data['f_spec_spider']
# normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
# window=400
# step=20
# data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
# all_web_ff = data['ff']
# all_web_stft = data['f_spec']
# f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
# coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
# roi_spider_list=coordinates['roi_spider_list']
# roi_fly_list=coordinates['roi_fly_list']
# data = np.load(fname.replace('.avi', '.xyt.npy'))
# roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# # t= [i for i in range(4000, 6000, 10)]
# roi_t = np.array(roi_t)
# roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
# t = [i for i in range(250, (data.shape[2] - 250), 125)]
# # t= [i for i in range(4000, 6000, 10)]
# t = np.array(t)
# fig = plt.figure()
# roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
# roi_f_idx = roi_f_idx[0]
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 0:40], axis=1), label='Courch 1')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 40:103], axis=1), label='Static after crouch')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 103:203], axis=1), label='Courch 2')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 203:241], axis=1), label='Shake')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 241:291], axis=1), label='Courch 3')
# plt.plot(roi_ff[roi_f_idx], np.mean(normalized_fly[roi_f_idx, 291:417], axis=1), label='Courch 3')
# plt.legend()
#
# plt.show()



#### Plot average results from all videos.
crouch  = np.array(Crouch)
shake = np.array(Shake)
shake2 = np.array(Shake2)
static=np.array(Static)
static_after_crouch = np.array(Static_after_crouch)
static_after_shake = np.array(Static_after_shake)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
plt.plot(roi_ff[roi_f_idx], np.mean(static, axis=0), label='Static')
plt.plot(roi_ff[roi_f_idx], np.mean(crouch, axis=0), label='Crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(shake, axis=0), label='Shake')
plt.plot(roi_ff[roi_f_idx], np.mean(static_after_crouch, axis=0), label='Static after crouch')
plt.plot(roi_ff[roi_f_idx], np.mean(shake2, axis=0), label='Shake before static')
plt.plot(roi_ff[roi_f_idx], np.mean(static_after_shake, axis=0), label='Static after shake')

plt.legend()
plt.show()

fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
[lower,upper] = st.t.interval(alpha=0.95, df=len(static.T) - 1,
              loc=np.mean(static, axis=0),
              scale=st.sem(static, axis=0))
plt.plot(roi_ff[roi_f_idx], np.mean(static, axis=0), label='Static')
# plt.fill_between(roi_ff[roi_f_idx], (lower), (upper),  alpha=0.5)
# [lower,upper] = st.t.interval(alpha=0.95, df=len(crouch.T) - 1,
#               loc=np.mean(crouch, axis=0),
#               scale=st.sem(crouch, axis=0))
plt.plot(roi_ff[roi_f_idx], np.mean(crouch, axis=0), label='Crouch')
# plt.fill_between(roi_ff[roi_f_idx], (lower), (upper),  alpha=0.5)
# [lower,upper] = st.t.interval(alpha=0.95, df=len(shake.T) - 1,
#               loc=np.mean(shake, axis=0),
#               scale=st.sem(shake, axis=0))
plt.plot(roi_ff[roi_f_idx], np.mean(shake, axis=0), label='Shake')
# plt.fill_between(roi_ff[roi_f_idx], (lower), (upper),  alpha=0.5)
# [lower,upper] = st.t.interval(alpha=0.95, df=len(static_after_crouch.T) - 1,
#               loc=np.mean(static_after_crouch, axis=0),
#               scale=st.sem(static_after_crouch, axis=0))
plt.plot(roi_ff[roi_f_idx], np.mean(static_after_crouch, axis=0), label='Static after crouch')
# plt.fill_between(roi_ff[roi_f_idx], (lower), (upper),  alpha=0.5)
# [lower,upper] = st.t.interval(alpha=0.95, df=len(static_after_shake.T) - 1,
#               loc=np.mean(static_after_shake, axis=0),
#               scale=st.sem(static_after_shake, axis=0))
plt.plot(roi_ff[roi_f_idx], np.mean(static_after_shake, axis=0), label='Static after shake')
# plt.fill_between(roi_ff[roi_f_idx], (lower), (upper),  alpha=0.5)
plt.legend()
plt.show()



import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('Qt5Agg')
import scipy.stats as st
fname = 'Z:\HsinYi\web_vibration/110524/110524_spider_prey_d_200hz_0_10_t1_C001H001S0001/110524_spider_prey_d_200hz_0_10_t1_C001H001S0001.avi'
video_path = 'Z:\HsinYi\web_vibration/110524/110524_spider_prey_d_200hz_0_10_t1_C001H001S0001/110524 Spider Prey D 200Hz 0 10 T1 C001h001s0001.mp4'
data =np.load(fname.replace( '.avi','_roi_stft.npz'))
roi_ff = data['ff']
roi_f_spec_fly = data['f_spec_fly']
roi_f_spec_fly_p = data['f_spec_fly_p']
roi_f_spec_spider = data['f_spec_spider']
normalized_fly = np.divide(roi_f_spec_fly, roi_f_spec_fly_p)
window=400
step=20
data = np.load(fname.replace( '.avi','_stft_window500_shift125.npz'))
all_web_ff = data['ff']
all_web_stft = data['f_spec']
f_idx = np.where((all_web_ff >= 1) & (all_web_ff <= 50))
coordinates = np.load(video_path.replace('.mp4', "_roi_coordinates.npz"))
roi_spider_list=coordinates['roi_spider_list']
roi_fly_list=coordinates['roi_fly_list']
data = np.load(fname.replace('.avi', '.xyt.npy'))
roi_t = [i for i in range(int(window / 2), (data.shape[2] - int(window / 2)), step)]
# t= [i for i in range(4000, 6000, 10)]
roi_t = np.array(roi_t)
roi_f_idx = np.where((roi_ff >= 1) & (roi_ff <= 50))
t = [i for i in range(500, (data.shape[2] - 500), 10)]
# t= [i for i in range(4000, 6000, 10)]
t = np.array(t)
fig = plt.figure()
roi_f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
roi_f_idx = roi_f_idx[0]
f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
ax2 = plt.subplot()
ax2.pcolormesh(roi_t, roi_ff[f_idx[0]], roi_f_spec_fly_p[f_idx[0], :],vmin=0, vmax=2000)

fig = plt.figure()
f_idx = np.where((roi_ff >= 0) & (roi_ff <= 500))
ax2 = plt.subplot()
ax2.pcolormesh(roi_t, roi_ff[f_idx[0]], normalized_fly[f_idx[0], :],vmin=0, vmax=2)
