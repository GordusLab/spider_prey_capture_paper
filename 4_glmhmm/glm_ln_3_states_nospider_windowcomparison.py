
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
f1_250=[]

for shuffle in range(250):
    accuracy_list=[]
    accuracy_list2=[]
    accuracy_list22=[]
    accuracy_list4=[]
    states_linear_output=[]
    states_linear_output2=[]
    states_linear_output22=[]
    states_linear_output4=[]
    sta_all=[]
    sta_vx_all=[]
    ### time window 1000 only has 28591 data points rather than 42991
    shuffled_indicies = np.random.permutation(42751)
    train_data_size = int(42751 * 3 / 4)
    for stimulus_window in range(25,26,5):
        prob_states_mul_lr=[]
        prob_states_lr = []
        prob_states_lr_2=[]
        prob_states_lr_22 = []
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




            ## 2 features

            train_x = np.array(linear_output_extend)[shuffled_indicies[:train_data_size]]
            train_x_fly = np.array(linear_output_vx_extend)[shuffled_indicies[:train_data_size]]
            train_y = np.array(labeled_beh_extend)[shuffled_indicies[:train_data_size]]
            test_x = np.array(linear_output_extend)[shuffled_indicies[train_data_size:]]
            test_x_fly = np.array(linear_output_vx_extend)[shuffled_indicies[train_data_size:]]
            # test_x = np.array(observation_extend)[shuffled_indicies[train_data_size:]]
            test_y = np.array(labeled_beh_extend)[shuffled_indicies[train_data_size:]]
            lr = linear_model.LogisticRegression()
            lr.fit(np.append(train_x, train_x_fly, axis=0).reshape(2, len(train_x)).transpose(), train_y)
            mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(
                np.append(train_x, train_x_fly, axis=0).reshape(2, len(train_x)).transpose(), train_y)

            test_x_prob_lr = lr.predict_proba(np.append(test_x, test_x_fly, axis=0).reshape(2, len(test_x)).transpose())
            prob_states_lr_2.append(test_x_prob_lr)
            states_linear_output2.append(np.array(np.append(test_x, test_x_fly, axis=0).reshape(2, len(test_x)).transpose()))


        prob_temp_2 = np.array(prob_states_lr_2)
        prob_temp_2 = np.swapaxes(prob_temp_2, 0, 1)

        idx_list_2 = []
        for i in range(len(prob_temp_2)):
            idx_list_2.append(np.where(prob_temp_2[i,:] == np.max(prob_temp_2[i,:]))[1][0])
        idx_list_2 = np.array(idx_list_2)
        accuracy = sum((test_y == np.array(idx_list_2).astype(int)).astype(int)) / len(idx_list_2)
        # print("Two features accuracy = ",accuracy)
        # print('Two features F score = ', f1_score(test_y, np.array(idx_list_2).astype(int), average ='micro'))
        accuracy_list2.append(accuracy)


        count=train_data_size
        time = 0
        time_count=[]
        for i in range(len(labeled_beh)):
            time += len(labeled_beh[i])
            time_count.append(time)
        if stimulus_window==25:
            for i in range(4):
                if i==0:
                    f1_250.append(f1_score(test_y[0: time_count[12]-count], np.array(idx_list_2[0: time_count[12]-count]).astype(int), average='micro'))

                else:
                    f1_250.append(f1_score(test_y[time_count[i+11]-count: time_count[i+12]-count], np.array(idx_list_2[time_count[i+11]-count: time_count[i+12]-count]).astype(int),
                                         average='micro'))
        # elif stimulus_window == 10:
        #     for i in range(4):
        #         if i == 0:
        #             f1_100.append(f1_score(test_y[0: time_count[12] - count],
        #                                   np.array(idx_list_2[0: time_count[12] - count]).astype(int), average='micro'))
        #
        #         else:
        #             f1_100.append(f1_score(test_y[time_count[i + 11] - count: time_count[i + 12] - count],
        #                                   np.array(idx_list_2[time_count[i + 11] - count: time_count[i + 12] - count]).astype(
        #                                       int),
        #                                   average='micro'))
        # elif stimulus_window == 15:
        #     for i in range(4):
        #         if i == 0:
        #             f1_150.append(f1_score(test_y[0: time_count[12] - count],
        #                                   np.array(idx_list_2[0: time_count[12] - count]).astype(int), average='micro'))
        #
        #         else:
        #             f1_150.append(f1_score(test_y[time_count[i + 11] - count: time_count[i + 12] - count],
        #                                   np.array(idx_list_2[time_count[i + 11] - count: time_count[i + 12] - count]).astype(
        #                                       int),
        #                                   average='micro'))
        # if stimulus_window == 20:
        #     for i in range(4):
        #         if i == 0:
        #             f1_200.append(f1_score(test_y[0: time_count[12] - count],
        #                                   np.array(idx_list_2[0: time_count[12] - count]).astype(int), average='micro'))
        #
        #         else:
        #             f1_200.append(f1_score(test_y[time_count[i + 11] - count: time_count[i + 12] - count],
        #                                   np.array(idx_list_2[time_count[i + 11] - count: time_count[i + 12] - count]).astype(
        #                                       int),
        #                                   average='micro'))

np.save('F1s_250ms.npy', np.array(f1_250))
# np.save('F1s_100ms.npy', np.array(f1_100))
# np.save('F1s_150ms.npy', np.array(f1_150))
# np.save('F1s_200ms.npy', np.array(f1_200))
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Example LL data for 3, 4, and 5 states (replace with your actual data)
#
# # Combine into one array
# data = [f1_50, f1_100, f1_150, f1_200]
# labels = ['50ms', '100ms', '150ms', '200ms']
#
# # Scatter plot
# plt.figure(figsize=(8, 5))
#
# for i, ll in enumerate(data):
#     x = np.random.normal(i , 0.05, size=len(ll))  # jitter for visibility
#     plt.scatter(x, ll, alpha=0.6, label=labels[i], c='darkgray')
#
# data = np.array(data)
# data = np.transpose(data)
# mean_bic = np.mean(data, axis=0)
# std = np.std(data, axis=0, ddof=1)
# states = np.arange(0, 0 + len(mean_bic))
# plt.errorbar(states, mean_bic, yerr=std, fmt='-o', color='black', capsize=5)
#
# # Customize axes
# plt.xticks([0, 1, 2, 3], labels)
# plt.ylabel('F1')
# plt.ylim([0.5, 0.9])
#
# plt.tight_layout()
# plt.show()
#
#

#
