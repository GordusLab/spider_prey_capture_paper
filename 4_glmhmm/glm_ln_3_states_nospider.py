
import glob
import matplotlib
matplotlib.use('TkAgg')
from sklearn import linear_model
from scipy.signal import hilbert
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
#stimulus_window = 70  #stimulus length for STA (stimulus trigger ensembles average)
roll_time = 5  #rolling time for probility estimation of the outcome
states_label=[0,1,2]
v_smooth_window=2
lp_f = 25
accuracy_list=[]
accuracy_list2=[]
accuracy_list4=[]
states_linear_output=[]
states_linear_output2=[]
states_linear_output4=[]
fig, axs = plt.subplots(1,2)
sta_all=[]
sta_vx_all=[]
shuffled_indicies = np.random.permutation(42991)
train_data_size = int(42991* 3 / 4)
for stimulus_window in range(10,11,2):
    print(stimulus_window)
    prob_states_mul_lr=[]
    prob_states_lr = []
    prob_states_lr_2=[]
    prob_states_lr_4 = []
    v_x_all =[]
    v_y_all=[]
    for states in states_label:
        #initial_time = [569, 276, 99,184,1776,567,2435,226,1017]
        initial_time = [569, 276, 99,1776,  567, 2435,  141, 109, 204, 400, 377, 206, 170,177, 207, 169]

        videodirectory = 'B:/HsinYi/GitHub/sensory_beh/'
        videoname = '*'
        sensory_dic = videodirectory+'sensory_auto/'
        sensory_roi_files = glob.glob(sensory_dic+ videoname+ '_roi_coordinates_side.npz')
        sensory_stft_files = glob.glob(sensory_dic+ videoname+ '_roi_stft_side.npz')

        labeled_beh = []
        observations = []
        stimulus =[]
        stimulus_all=[]
        stimulus_fly=[]
        stimulus_v_x = []
        stimulus_v_y = []
        stimulus_fly_all=[]
        stimulus_v_x_all = []
        stimulus_v_y_all = []
        observations_fly = []
        for j in range(len(sensory_roi_files)):
            sensory_roi = np.load(sensory_roi_files[j])
            sensory_stft = np.load(sensory_stft_files[j])

            roi_fly_list = sensory_roi['roi_fly_list']
            roi_spider_list = sensory_roi['roi_spider_list']
            nframes = roi_fly_list.shape[0]
            ff_stft = sensory_stft['ff']
            f_spec_fly = sensory_stft['f_spec_fly']
            f_spec_fly_p = sensory_stft['f_spec_fly_p']
            f_spec_spider_p = sensory_stft['f_spec_spider_p']
            vid_name = sensory_roi_files[j].split('/sensory_auto\\')[1].split('_roi_coordinates_side.npz')[0]
            r = roi_spider_list[:, 1] + roi_spider_list[:, 3] / 2 - roi_fly_list[:, 1] - roi_fly_list[:, 3] / 2
            r = np.abs(r)
            r = r / ((roi_spider_list[:, 2]+roi_spider_list[:, 3]))/2
            #r = r / np.min(r)
            r = (1 / r) ** 2
            r = r / np.max(r)
            r[np.where(r<0.5)[0]]=0.5
            # r=r+0.5
            # r=r/np.max(r)
            x = roi_fly_list[:, 0] + roi_fly_list[:, 2] / 2
            v_x = x - np.roll(x, 1)
            v_x = v_x[1:len(v_x)]
            y = roi_fly_list[:, 1] + roi_fly_list[:, 3] / 2
            v_y = y - np.roll(y, 1)
            v_y = v_y[1:len(v_y)]
            v_x = hilbert(v_x)
            v_x = np.abs(v_x)
            v_y = hilbert(v_y)
            v_y = np.abs(v_y)

            ## Load beh
            # filenames = glob.glob(videodirectory+ 'wavelet/'+vid_name+ '*_croprotaligned01234_nonormalized_wavelet.npz')
            # csv_filenames = glob.glob(videodirectory+'merging_static/' + vid_name + '*_wavelet_timestep.csv')
            beh_hmm = np.load(
                glob.glob(videodirectory + 'sensory_auto/' + vid_name + '*_hmm_umap_filtered_predictedlabels.npy')[0]
                )

            ## Sensory
            test_divide = np.divide(f_spec_fly, f_spec_fly_p)
            # test_divide = np.mean(test_divide, axis=0)
            t = [i for i in range(int(40 / 2), (nframes - int(40 / 2)), 2)]
            # t= [i for i in range(4000, 6000, 10)]
            # if '0328' in sensory_roi_files[j]:
            #     t.append(5100)
            t = np.array(t)

            t3 = [i for i in range(0, initial_time[j], 1)]
            t3 = np.array(t3)
            yinterp3 = np.interp(t3, t, np.mean(test_divide[0:13, ], axis=0))
            yinterp3_mean = np.mean(yinterp3)
            yinterp3_std = np.std(yinterp3)
            yinterp3fly = np.interp(t3, t, np.mean(f_spec_fly[0:13, ], axis=0))
            yinterp3fly_mean = np.mean(yinterp3fly)
            yinterp3fly_std = np.std(yinterp3fly)
            yinterp3spider = np.interp(t3, t, np.mean(f_spec_spider_p[0:13, ], axis=0))
            yinterp3spider_mean = np.mean(yinterp3spider)
            yinterp3spider_std = np.std(yinterp3spider)

            t2 = [i for i in range(initial_time[j], (len(beh_hmm) + initial_time[j]), 1)]
            # # t= [i for i in range(4000, 6000, 10)]
            t2 = np.array(t2)
            #
            yinterp = np.interp(t2, t, np.mean(test_divide[0:13, ], axis=0))
            yinterp = (yinterp - yinterp3_mean) / yinterp3_std
            yinterpspider = np.interp(t2, t, np.mean(f_spec_spider_p[0:13, ], axis=0))
            #yinterpspider = (yinterpspider - yinterp3spider_mean) / yinterp3spider_std
            yinterpspider = yinterpspider / (np.max(yinterpspider))
            # yinterp = yinterp/(np.max(yinterp))
            # yinterp = interp1d(t,  test_divide, axis=1)
            # yinterp = yinterp(t2)
            yinterpfly = np.interp(t2, t, np.mean(f_spec_fly[0:13, :], axis=0))
            yinterpfly = (yinterpfly - yinterp3fly_mean) / yinterp3fly_std
            #yinterpfly = yinterpfly / (np.max(yinterpfly))
            #yinterpfly = yinterpfly - yinterpspider * r[t2]
            #yinterpfly = yinterpfly / np.max(yinterpfly)
            # if states ==0:
            #
            #     v_x_all.append(v_x[t2][0:len(v_x[t2])-stimulus_window])
            #     v_y_all.append(v_y[t2][0:len(v_y[t2])-stimulus_window])
            v_x_sensory = v_x[t2][0:len(v_x[t2])]
            v_y_sensory = v_y[t2][0:len(v_y[t2])]

            sos = signal.butter(10, lp_f, 'lp', fs=100, output='sos')
            v_x_sensory = signal.sosfilt(sos, v_x_sensory)
            v_y_sensory = signal.sosfilt(sos, v_y_sensory)

            v_x_sensory = (v_x_sensory  -np.mean(v_x_sensory )) / np.std(v_x_sensory)
            v_y_sensory = (v_y_sensory  -np.mean(v_y_sensory )) / np.std(v_y_sensory)

            # for v_time in range(len(v_x_sensory)):
            #     if v_time < int(v_smooth_window / 2):
            #         v_x_sensory[v_time] = np.max(v_x_sensory[0: int(v_smooth_window)])
            #         v_y_sensory[v_time] = np.max(v_y_sensory[0: int(v_smooth_window)])
            #     else:
            #         v_x_sensory[v_time] = np.max(
            #             v_x_sensory[v_time - int(v_smooth_window / 2): v_time + int(v_smooth_window / 2)])
            #         v_y_sensory[v_time] = np.max(
            #             v_y_sensory[v_time - int(v_smooth_window / 2): v_time + int(v_smooth_window / 2)])

            ## Spike trigger average
            from itertools import groupby
            count_dups = [sum(1 for _ in group) for _, group in groupby(beh_hmm)]
            cumsum = np.cumsum(count_dups)

            groups = []
            uniquekeys = []
            for k, g in groupby(beh_hmm):
                groups.append(list(g))  # Store group iterator as a list
                uniquekeys.append(k)

            idx = np.where(np.array(uniquekeys)==states)
            stimulus_temp=[]
            stimulus_temp_fly=[]

            stimulus_v_x_temp = []
            stimulus_v_y_temp = []

            for trigger in range(len(idx[0])):
                if idx[0][trigger]==0:
                    continue
                else:
                    event_time = cumsum[idx[0][trigger]-1]
                    if len(yinterp[event_time - stimulus_window:event_time])==0:
                        stimulus.append(yinterp[0:stimulus_window])
                        stimulus_fly.append(yinterpfly[0:stimulus_window])
                        stimulus_temp.append(yinterp[0:stimulus_window])
                        stimulus_temp_fly.append(yinterpfly[0:stimulus_window])
                        stimulus_v_x.append(v_x_sensory[0:stimulus_window])
                        stimulus_v_y.append(v_y_sensory[0:stimulus_window])
                        stimulus_v_x_temp.append(v_x_sensory[0:stimulus_window])
                        stimulus_v_y_temp.append(v_y_sensory[0:stimulus_window])
                    else:
                        stimulus.append(yinterp[event_time - stimulus_window:event_time])
                        stimulus_fly.append(yinterpfly[event_time - stimulus_window:event_time])
                        stimulus_temp.append(yinterp[event_time - stimulus_window:event_time])
                        stimulus_temp_fly.append(yinterpfly[event_time - stimulus_window:event_time])
                        stimulus_v_x.append(v_x_sensory[event_time - stimulus_window:event_time])
                        stimulus_v_y.append(v_y_sensory[event_time - stimulus_window:event_time])
                        stimulus_v_x_temp.append(v_x_sensory[event_time - stimulus_window:event_time])
                        stimulus_v_y_temp.append(v_y_sensory[event_time - stimulus_window:event_time])


            stimulus_all.append(stimulus_temp)
            stimulus_fly_all.append(stimulus_temp_fly)
            stimulus_v_x_all.append(stimulus_v_x_temp)
            stimulus_v_y_all.append(stimulus_v_y_temp)


            #beh_hmm= np.load(filenames[0].replace('.npz', '_hmm_umap_filtered_predictedlabels.npy'))
            #beh_manual= np.load(filenames[0].replace('.npz', '_manuallabels.npy'))




            labeled_beh.append(beh_hmm)
            observations.append(list(yinterp))
            observations_fly.append(list(yinterpfly))


        ## Spike trigger average of all videos. Mean centered
        stimulus_test = np.array(stimulus)
        data_ensembles_len = stimulus_test.shape[0]
        arr = np.arange(data_ensembles_len)
        np.random.shuffle(arr)
        #train_set = stimulus_test[arr[0:int(len(arr)*3/4)]]
        #test_set = stimulus_test[arr[int(len(arr)*3/4):len(arr)]]
        train_set = stimulus_test
        sta = np.mean(train_set, axis=0)
        #sta = sta-np.mean(sta)



        stimulus_test = np.array(stimulus_fly)
        data_ensembles_len = stimulus_test.shape[0]
        arr = np.arange(data_ensembles_len)
        np.random.shuffle(arr)
        # train_set = stimulus_test[arr[0:int(len(arr)*3/4)]]
        # test_set = stimulus_test[arr[int(len(arr)*3/4):len(arr)]]
        train_set = stimulus_test
        sta_fly = np.mean(train_set, axis=0)
        #sta_fly = sta_fly-np.mean(sta_fly)

        stimulus_test = np.array(stimulus_v_x)
        data_ensembles_len = stimulus_test.shape[0]
        arr = np.arange(data_ensembles_len)
        np.random.shuffle(arr)
        # train_set = stimulus_test[arr[0:int(len(arr)*3/4)]]
        # test_set = stimulus_test[arr[int(len(arr)*3/4):len(arr)]]
        train_set = stimulus_test
        sta_vx = np.mean(train_set, axis=0)

        stimulus_test = np.array(stimulus_v_y)
        data_ensembles_len = stimulus_test.shape[0]
        arr = np.arange(data_ensembles_len)
        np.random.shuffle(arr)
        # train_set = stimulus_test[arr[0:int(len(arr)*3/4)]]
        # test_set = stimulus_test[arr[int(len(arr)*3/4):len(arr)]]
        train_set = stimulus_test
        sta_vy = np.mean(train_set, axis=0)

        axs[0].plot(sta, label=str(states))
        sta_all.append(sta)
        axs[0].legend()
        # axs[1].plot(sta_fly, label='sta_fly'+str(states))
        # axs[1].legend()
        axs[1].plot(sta_vx, label=str(states))
        sta_vx_all.append(sta_vx)
        axs[1].legend()
        # axs[3].plot(sta_vy, label=str(states))
        # axs[3].legend()




        ### Step 2: Linear Output

        labeled_beh = []
        observations = []
        observations_fly = []

        linear_output_all=[]
        linear_output_all_fly = []
        linear_output_all_vx = []
        linear_output_all_vy = []
        for j in range(len(sensory_roi_files)):
            sensory_roi = np.load(sensory_roi_files[j])
            sensory_stft = np.load(sensory_stft_files[j])
            roi_fly_list = sensory_roi['roi_fly_list']
            roi_spider_list = sensory_roi['roi_spider_list']
            nframes = roi_fly_list.shape[0]
            ff_stft = sensory_stft['ff']
            f_spec_fly = sensory_stft['f_spec_fly']
            f_spec_fly_p = sensory_stft['f_spec_fly_p']
            f_spec_spider_p = sensory_stft['f_spec_spider_p']
            vid_name = sensory_roi_files[j].split('/sensory_auto\\')[1].split('_roi_coordinates_side.npz')[0]
            r = roi_spider_list[:, 1] + roi_spider_list[:, 3] / 2 - roi_fly_list[:, 1] - roi_fly_list[:, 3] / 2
            r = np.abs(r)
            r = r / ((roi_spider_list[:, 2] + roi_spider_list[:, 3])) / 2
            # r = r / np.min(r)
            r = (1 / r) ** 2
            r = r / np.max(r)
            r[np.where(r < 0.5)[0]] = 0.5
            # r = r + 0.5
            # r = r / np.max(r)
            x = roi_fly_list[:, 0] + roi_fly_list[:, 2] / 2
            v_x = x - np.roll(x, 1)
            v_x = v_x[1:len(v_x)]
            y = roi_fly_list[:, 1] + roi_fly_list[:, 3] / 2
            v_y = y - np.roll(y, 1)
            v_y = v_y[1:len(v_y)]
            v_x = hilbert(v_x)
            v_x = np.abs(v_x)
            v_y = hilbert(v_y)
            v_y = np.abs(v_y)

            ## Load beh
            # filenames = glob.glob(videodirectory+ 'wavelet/'+vid_name+ '*_croprotaligned01234_nonormalized_wavelet.npz')
            # csv_filenames = glob.glob(videodirectory+'merging_static/' + vid_name + '*_wavelet_timestep.csv')
            beh_hmm = np.load(
                glob.glob(videodirectory + 'sensory_auto/' + vid_name + '*_hmm_umap_filtered_predictedlabels.npy')[0]
            )

            ## Sensory
            test_divide = np.divide(f_spec_fly, f_spec_fly_p)
            # test_divide = np.mean(test_divide, axis=0)
            t = [i for i in range(int(40 / 2), (nframes - int(40 / 2)), 2)]
            # t= [i for i in range(4000, 6000, 10)]
            # if '0328' in sensory_roi_files[j]:
            #     t.append(5100)
            t = np.array(t)

            t3 = [i for i in range(0, initial_time[j], 1)]
            t3 = np.array(t3)
            yinterp3 = np.interp(t3, t, np.mean(test_divide[0:13, ], axis=0))
            yinterp3_mean = np.mean(yinterp3)
            yinterp3_std = np.std(yinterp3)
            yinterp3fly = np.interp(t3, t, np.mean(f_spec_fly[0:13, ], axis=0))
            yinterp3fly_mean = np.mean(yinterp3fly)
            yinterp3fly_std = np.std(yinterp3fly)
            yinterp3spider = np.interp(t3, t, np.mean(f_spec_spider_p[0:13, ], axis=0))
            yinterp3spider_mean = np.mean(yinterp3spider)
            yinterp3spider_std = np.std(yinterp3spider)

            t2 = [i for i in range(initial_time[j], (len(beh_hmm) + initial_time[j]), 1)]
            # # t= [i for i in range(4000, 6000, 10)]
            t2 = np.array(t2)
            #
            yinterp = np.interp(t2, t, np.mean(test_divide[0:13, ], axis=0))
            yinterp = (yinterp - yinterp3_mean) / yinterp3_std
            yinterpspider = np.interp(t2, t, np.mean(f_spec_spider_p[0:13, ], axis=0))
            #yinterpspider = (yinterpspider - yinterp3spider_mean) / yinterp3spider_std
            yinterpspider= yinterpspider / (np.max(yinterpspider))
            # yinterp = yinterp/(np.max(yinterp))
            # yinterp = interp1d(t,  test_divide, axis=1)
            # yinterp = yinterp(t2)
            yinterpfly = np.interp(t2, t, np.mean(f_spec_fly[0:13, :], axis=0))
            yinterpfly = (yinterpfly - yinterp3fly_mean) / yinterp3fly_std
            #yinterpfly = yinterpfly/(np.max(yinterpfly))
            #yinterpfly = yinterpfly - yinterpspider * r[t2]

            v_x_sensory = v_x[t2][0:len(v_x[t2])]
            v_y_sensory = v_y[t2][0:len(v_y[t2])]

            sos = signal.butter(10, lp_f, 'lp', fs=100, output='sos')
            v_x_sensory = signal.sosfilt(sos, v_x_sensory)
            v_y_sensory = signal.sosfilt(sos, v_y_sensory)

            v_x_sensory = (v_x_sensory - np.mean(v_x_sensory)) / np.std(v_x_sensory)
            v_y_sensory = (v_y_sensory - np.mean(v_y_sensory)) / np.std(v_y_sensory)

            # for v_time in range(len(v_x_sensory)):
            #     if v_time < int(v_smooth_window / 2):
            #         v_x_sensory[v_time] = np.max(v_x_sensory[0: int(v_smooth_window)])
            #         v_y_sensory[v_time] = np.max(v_y_sensory[0: int(v_smooth_window)])
            #     else:
            #         v_x_sensory[v_time] = np.max(
            #             v_x_sensory[v_time - int(v_smooth_window / 2): v_time + int(v_smooth_window / 2)])
            #         v_y_sensory[v_time] = np.max(
            #             v_y_sensory[v_time - int(v_smooth_window / 2): v_time + int(v_smooth_window / 2)])

            linear_output=[]
            linear_output_fly=[]
            linear_output_v_x=[]
            linear_output_v_y=[]
            for k in range(len(yinterp)-stimulus_window):

                linear_output.append(np.dot(yinterp[k:(k+stimulus_window)],sta))
                linear_output_fly.append(np.dot(yinterpfly[k:(k + stimulus_window)], sta_fly))
                linear_output_v_x.append(np.dot(v_x_sensory[k:(k + stimulus_window)], sta_vx))
                linear_output_v_y.append(np.dot(v_y_sensory[k:(k + stimulus_window)], sta_vy))
                # linear_output_v_x.append(v_x_sensory[k]))
                # linear_output_v_y.append(v_y_sensory[k]))
            linear_output = np.array(linear_output)
            linear_output_fly = np.array(linear_output_fly)
            linear_output_v_x = np.array(linear_output_v_x)
            linear_output_v_y = np.array(linear_output_v_y)
            linear_output_all.append(linear_output)
            linear_output_all_fly.append(linear_output_fly)
            linear_output_all_vx.append(linear_output_v_x)
            linear_output_all_vy.append(linear_output_v_y)
            labeled_beh.append(beh_hmm[stimulus_window:len(beh_hmm)])

        linear_output_extend=[]
        linear_output_fly_extend = []
        linear_output_vx_extend = []
        linear_output_vy_extend = []
        labeled_beh_extend=[]
        for i in range(len(linear_output_all)):
            linear_output_extend.extend(linear_output_all[i])
            linear_output_fly_extend.extend(linear_output_all_fly[i])
            linear_output_vx_extend.extend(linear_output_all_vx[i])
            linear_output_vy_extend.extend(linear_output_all_vy[i])
            labeled_beh_extend.extend(labeled_beh[i])

        linear_output_sort = np.sort(linear_output_extend)
        ### Step 3: Characterize nonlinearity
        # for i in range(len(linear_output_extend)):
        states_linear_output.append(np.array(linear_output_extend))


        train_x = np.array(linear_output_extend)[shuffled_indicies[:train_data_size]]
        train_y = np.array(labeled_beh_extend)[shuffled_indicies[:train_data_size]]
        test_x = np.array(linear_output_extend)
        # test_x = np.array(observation_extend)[shuffled_indicies[train_data_size:]]
        test_y = np.array(labeled_beh_extend)

        ##Train multi-classification model with logistic regression

        lr = linear_model.LogisticRegression()
        lr.fit(np.array(train_x).reshape(-1, 1), train_y)
        test_x_prob_lr = lr.predict_proba(np.array(test_x).reshape(-1, 1))
        prob_states_lr.append(test_x_prob_lr)

        # Train multinomial logistic regression model
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(
            np.array(train_x).reshape(-1, 1), train_y)
        test_x_prob_mul_lr = mul_lr.predict_proba(np.array(test_x).reshape(-1, 1))
        prob_states_mul_lr.append(test_x_prob_mul_lr)


        ## 2 features

        train_x = np.array(linear_output_extend)[shuffled_indicies[:train_data_size]]
        train_x_fly = np.array(linear_output_vx_extend)[shuffled_indicies[:train_data_size]]
        train_y = np.array(labeled_beh_extend)[shuffled_indicies[:train_data_size]]
        test_x = np.array(linear_output_extend)
        test_x_fly = np.array(linear_output_vx_extend)
        # test_x = np.array(observation_extend)[shuffled_indicies[train_data_size:]]
        test_y = np.array(labeled_beh_extend)
        lr = linear_model.LogisticRegression()
        lr.fit(np.append(train_x, train_x_fly, axis=0).reshape(2, len(train_x)).transpose(), train_y)
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(
            np.append(train_x, train_x_fly, axis=0).reshape(2, len(train_x)).transpose(), train_y)

        test_x_prob_lr = lr.predict_proba(np.append(test_x, test_x_fly, axis=0).reshape(2, len(test_x)).transpose())
        prob_states_lr_2.append(test_x_prob_lr)
        states_linear_output2.append(np.array(np.append(test_x, test_x_fly, axis=0).reshape(2, len(test_x)).transpose()))

        ### Plot logistic regression
        # loss = test_x_prob_lr[:, 0]
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(test_x, test_x_fly, loss)
        # loss = test_x_prob_lr[:, 1]
        # ax.scatter(test_x, test_x_fly, loss, color='yellow')
        # loss = test_x_prob_lr[:, 2]
        # ax.scatter(test_x, test_x_fly, loss, color='orange')
        # ax.set_xlabel('Z score of fly STFT')
        # ax.set_ylabel('Fly y velocity')
        # ax.set_zlabel('Prob')
        # plt.show()

        ### 3 features

        train_x = np.array(linear_output_extend)[shuffled_indicies[:train_data_size]]
        train_x_fly = np.array(linear_output_fly_extend)[shuffled_indicies[:train_data_size]]
        train_x_vx = np.array(linear_output_vx_extend)[shuffled_indicies[:train_data_size]]
        train_x_vy = np.array(linear_output_vy_extend)[shuffled_indicies[:train_data_size]]
        train_y = np.array(labeled_beh_extend)[shuffled_indicies[:train_data_size]]
        test_x = np.array(linear_output_extend)
        test_x_fly = np.array(linear_output_fly_extend)
        test_x_vx = np.array(linear_output_vx_extend)
        test_x_vy = np.array(linear_output_vy_extend)
        # test_x = np.array(observation_extend)[shuffled_indicies[train_data_size:]]
        test_y = np.array(labeled_beh_extend)
        lr = linear_model.LogisticRegression()
        #temp = [train_x]+[train_x_fly]+[train_x_vx]+[train_x_vy]
        temp = [train_x] + [train_x_vx] + [train_x_vy]
        temp = np.array(temp)
        lr.fit(temp.transpose(), train_y)
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(
            temp.transpose(), train_y)
        #temp = [test_x] + [test_x_fly] + [test_x_vx] + [test_x_vy]
        temp = [test_x] + [test_x_vx] + [test_x_vy]
        temp = np.array(temp)
        test_x_prob_lr = lr.predict_proba(temp.transpose())
        prob_states_lr_4.append(test_x_prob_lr)
        states_linear_output4.append(np.array(temp.transpose()))


    prob_temp = np.array(prob_states_lr)
    prob_temp = np.swapaxes(prob_temp, 0, 1)

    idx_list = []
    for i in range(len(prob_temp)):
        idx_list.append(np.where(prob_temp[i,:] == np.max(prob_temp[i,:]))[1][0])
    idx_list =np.array(idx_list)
    accuracy = sum((test_y==np.array(idx_list).astype(int)).astype(int))/len(idx_list)
    print("One feature accuracy = ", accuracy)
    print('One feature F score = ', f1_score(test_y, np.array(idx_list).astype(int), average='micro'))
    accuracy_list.append(accuracy)

    prob_temp_2 = np.array(prob_states_lr_2)
    prob_temp_2 = np.swapaxes(prob_temp_2, 0, 1)

    idx_list_2 = []
    for i in range(len(prob_temp_2)):
        idx_list_2.append(np.where(prob_temp_2[i,:] == np.max(prob_temp_2[i,:]))[1][0])
    idx_list_2 = np.array(idx_list_2)
    accuracy = sum((test_y == np.array(idx_list_2).astype(int)).astype(int)) / len(idx_list_2)
    print("Two features accuracy = ",accuracy)
    print('Two features F score = ', f1_score(test_y, np.array(idx_list_2).astype(int), average ='micro'))
    accuracy_list2.append(accuracy)

    prob_temp_4 = np.array(prob_states_lr_4)
    prob_temp_4 = np.swapaxes(prob_temp_4, 0, 1)
    idx_list_4 = []
    for i in range(len(prob_temp_4)):
        idx_list_4.append(np.where(prob_temp_4[i,:] == np.max(prob_temp_4[i,:]))[1][0])
    idx_list_4 = np.array(idx_list_4)
    accuracy = sum((test_y == np.array(idx_list_4).astype(int)).astype(int)) / len(idx_list_4)
    print("Three features accuracy = ",accuracy)
    print('Three features F score = ', f1_score(test_y, np.array(idx_list_4).astype(int), average ='micro'))
    accuracy_list4.append(accuracy)


from itertools import groupby
spider_beh_predict = np.copy(np.array(idx_list_2))
count_dups = [sum(1 for _ in group) for _, group in groupby( np.array(idx_list_2))]
cumsum= np.cumsum(count_dups)
groups = []
uniquekeys = []
for k, g in groupby( np.array(idx_list_2)):
    groups.append(list(g))      # Store group iterator as a list
    uniquekeys.append(k)
idx = np.arange(0, len(uniquekeys))

count_loop=0
while len(np.where(np.array(count_dups)<15)[0])>0:
    count_loop += 1
    if count_loop > 100:
        break
    spider_beh_predict = np.copy(np.array(spider_beh_predict))
    count_dups = [sum(1 for _ in group) for _, group in groupby(np.array(spider_beh_predict))]
    cumsum = np.cumsum(count_dups)
    groups = []
    uniquekeys = []
    for k, g in groupby(np.array(spider_beh_predict)):
        groups.append(list(g))  # Store group iterator as a list
        uniquekeys.append(k)
    idx = np.arange(0, len(uniquekeys))
    for i in range(len(idx)):
        if count_dups[idx[i]] < 15:
            if idx[i] > 0 and (idx[i] + 1) > len(count_dups):
                if count_dups[idx[i] - 1] - count_dups[idx[i] + 1] > 0:
                    spider_beh_predict[cumsum[(idx[i] - 1)]:cumsum[idx[i]]] = uniquekeys[idx[i] - 1]
                else:
                    spider_beh_predict[cumsum[(idx[i] - 1)]:cumsum[idx[i]]] = uniquekeys[idx[i] + 1]
            else:
                if idx[i] == 0:
                    spider_beh_predict[0:cumsum[(idx[i] + 1)]] = uniquekeys[idx[i] + 1]
                else:
                    spider_beh_predict[cumsum[(idx[i] - 1)]:cumsum[idx[i]]] = uniquekeys[idx[i] - 1]
accuracy = sum((test_y == np.array(spider_beh_predict).astype(int)).astype(int)) / len(idx_list_2)
print(accuracy)
spider_beh_predict = np.copy(np.array(spider_beh_predict))
count_dups = [sum(1 for _ in group) for _, group in groupby(np.array(spider_beh_predict))]
cumsum = np.cumsum(count_dups)
groups = []
uniquekeys = []
for k, g in groupby(np.array(spider_beh_predict)):
    groups.append(list(g))  # Store group iterator as a list
    uniquekeys.append(k)
idx = np.arange(0, len(uniquekeys))

states_linear_output2_normalized = np.copy(states_linear_output2)
a = np.max(states_linear_output2[0][:, 0])
a2 = np.max(states_linear_output2[0][:, 1])
b = np.max(states_linear_output2[1][:, 0])
b2 = np.max(states_linear_output2[1][:, 1])
c = np.max(states_linear_output2[2][:, 0])
c2 = np.max(states_linear_output2[2][:, 1])
states_linear_output2_normalized[0][:, 0] = states_linear_output2_normalized[0][:, 0] / a
states_linear_output2_normalized[0][:, 1] = states_linear_output2_normalized[0][:, 1] / a2
states_linear_output2_normalized[1][:, 0] = states_linear_output2_normalized[1][:, 0] / b
states_linear_output2_normalized[1][:, 1] = states_linear_output2_normalized[1][:, 1] / b2
states_linear_output2_normalized[2][:, 0] = states_linear_output2_normalized[2][:, 0] / c
states_linear_output2_normalized[2][:, 1] = states_linear_output2_normalized[2][:, 1] / c2
states_linear_output4_normalized = np.copy(states_linear_output4)
a = np.max(states_linear_output4[0][:, 0])
a2 = np.max(states_linear_output4[0][:, 1])
b = np.max(states_linear_output4[1][:, 0])
b2 = np.max(states_linear_output4[1][:, 1])
c = np.max(states_linear_output4[2][:, 0])
c2 = np.max(states_linear_output4[2][:, 1])
states_linear_output4_normalized[0][:, 0] = states_linear_output4_normalized[0][:, 0] / a
states_linear_output4_normalized[0][:, 1] = states_linear_output4_normalized[0][:, 1] / a2
states_linear_output4_normalized[1][:, 0] = states_linear_output4_normalized[1][:, 0] / b
states_linear_output4_normalized[1][:, 1] = states_linear_output4_normalized[1][:, 1] / b2
states_linear_output4_normalized[2][:, 0] = states_linear_output4_normalized[2][:, 0] / c
states_linear_output4_normalized[2][:, 1] = states_linear_output4_normalized[2][:, 1] / c2
lo = []
lo_normalized = []
lo4 = []
lo4_normalized = []
for i in range(len(labeled_beh_extend)):
    lo.append(states_linear_output2[idx_list_2[i]][i])
    lo_normalized.append(states_linear_output2_normalized[idx_list_2[i]][i])
    lo4.append(states_linear_output4[idx_list_4[i]][i])
    lo4_normalized.append(states_linear_output4_normalized[idx_list_4[i]][i])

np.save('lo.npy', lo)
np.save('lo_normalized.npy', lo_normalized)
np.save('lo4_normalized.npy', lo4_normalized)
np.save('lo4.npy', lo4)
np.savez('states_linear_output_4features.npz', states_linear_output4=states_linear_output4,
         labeled_beh_extend=labeled_beh_extend, prob_states_lr_4=prob_states_lr_4)
np.savez('states_linear_output_2features.npz', states_linear_output2=states_linear_output2,
         labeled_beh_extend=labeled_beh_extend, prob_states_lr_2=prob_states_lr_2)


### Fly veolocity FFT
# import scipy
# dataFFT = []
# ff=[]
# for i in range(len(linear_output_all_vx)):
#     dataFFT.append(np.abs(scipy.fft.fft(linear_output_all_vx[i])))
#     ff.append(np.fft.fftfreq(len(linear_output_all_vx[i]), 0.01))
#
#
#
# for i in range(16):
#     dataFFT[i]  = dataFFT[i][ff[i]>0]
#     ff[i] = ff[i][ff[i]>0]  # Fill each row with Z[i]
#
#
# max_len = max(len(x) for x in ff)
# Z_combined = np.zeros((len(ff), max_len))  # Initialize with zeros
# # Align each Z[i] with the corresponding X[i]
# for i, z in enumerate(dataFFT):
#     Z_combined[i, :len(z)] = z  # Fill each row with Z[i]
#
# X_combined = np.linspace(0, 1, max_len)
# Y_combined = np.arange(len(ff) )  # Row indices (0, 1, 2, ..., 4)
#
# max_len = max(len(x) for x in ff)
# Z_combined = np.zeros((len(ff), max_len))  # Initialize with zeros
# # Align each Z[i] with the corresponding X[i]
# for i, z in enumerate(dataFFT):
#     Z_combined[i, :len(z)] = z  # Fill each row with Z[i]
#
# X_combined = np.linspace(0, 50, max_len)
# Y_combined = np.arange(len(ff) )  # Row indices (0, 1, 2, ..., 4)
#
#
# # Plot
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(X_combined, Y_combined, Z_combined, shading='nearest', cmap='viridis', vmax=500)
# plt.colorbar(label="Z values")
# plt.xlabel("Frequency")
# plt.ylabel("Recording number")
# plt.title("FFT of fly y velocity from side camera recording")
# plt.show()