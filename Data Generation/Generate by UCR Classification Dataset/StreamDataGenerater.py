import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from _sbd import _sbd
from _wdtw import _wdtw
from dtw import dtw
from _kmeans import kmeans
from joblib import Parallel, delayed
import time, math
 

def DataGenerator(X, y, n_class, class_choosen, healthy_ratio = 0.5, order_anomaly = [-1,-2,-3], save_path = None):

    data_ls_np = np.array(X)
    label_ls_np = np.array(y)

    time_series_not_choosen_idx = np.squeeze(np.argwhere(label_ls_np!=class_choosen))
    time_series_not_choosen_ls = data_ls_np[time_series_not_choosen_idx,:].tolist()

    centroids, cluster, label_km = kmeans(X, k=n_class)

    Association_matrix = np.zeros((n_class+1,n_class+1))
    for idx__ in range(len(y)): 
        Association_matrix[y[idx__]][label_km[idx__]] = Association_matrix[y[idx__]][label_km[idx__]] + 1

    time_series_base_ls = []

    for idx__ in range(len(y)): 
        if y[idx__] == class_choosen:
            if  label_km[idx__] == np.argmax(Association_matrix,axis=1)[class_choosen]:
                time_series_base_ls.append(data_ls[idx__])

    sbd_dis_ls = []
    for i in time_series_not_choosen_ls:
        sbd_dis_ls_ = []
        for j in time_series_base_ls:
            sbd_dis_ls_.append(_sbd(i,j)[0])
        sbd_dis_ls.append(np.mean(sbd_dis_ls_))

    time_series_anomaly_ls = []
    for i in order_anomaly:
        time_series_anomaly_ls.append(time_series_not_choosen_ls[np.argsort(sbd_dis_ls)[i]])

    print("We selected %s data as positive samples."%(len(time_series_base_ls)))
    print("We selected %s data as nagetive samples."%(len(time_series_anomaly_ls)))

    idx_time_series_base_ls = list(range(len(time_series_base_ls)))
    random.shuffle(idx_time_series_base_ls)
    time_series_train_data = [time_series_base_ls[i] for i in idx_time_series_base_ls[0:int(len(time_series_base_ls)*healthy_ratio)]]
    time_series_train_label = [0] * len(time_series_train_data) * time_series_base_ls[0].shape[0]
    time_series_test_data = [time_series_base_ls[i] for i in idx_time_series_base_ls[int(len(time_series_base_ls)*healthy_ratio):]] + time_series_anomaly_ls
    time_series_test_label = [0] * (len(time_series_test_data) - len(time_series_anomaly_ls)) + [1] * len(time_series_anomaly_ls)
            
    idx_time_series_test_ls = list(range(len(time_series_test_data)))
    random.shuffle(idx_time_series_test_ls)
    time_series_test_data = [time_series_test_data[i] for i in idx_time_series_test_ls]
    time_series_test_label = [time_series_test_label[i] for i in idx_time_series_test_ls]
    time_series_test_label_ = []
    for i in time_series_test_label:
        if i == 0 :
            time_series_test_label_.append([0]* time_series_base_ls[0].shape[0])
        if i == 1 :
            time_series_test_label_.append([1]*np.array(time_series_anomaly_ls[0]).shape[0])
    time_series_train_label = np.array(time_series_train_label).flatten()
    time_series_test_label = np.array(time_series_test_label_).flatten()

    time_series_data = np.array(time_series_train_data+time_series_test_data).flatten()
    time_series_label = np.concatenate((time_series_train_label,time_series_test_label),axis=0)

    if save_path != None:
        # print(save_path)
        save_path_name = 'ClassChoosen_%s_OrderAnoaly_%s_HealthyRatio_%s_'%(class_choosen,order_anomaly,healthy_ratio) + save_path[1]
        save_path_data = os.path.join(save_path[0], save_path_name + '_data.npy' )
        save_path_label = os.path.join(save_path[0], save_path_name + '_label.npy' )
        np.save(save_path_data,time_series_data)
        np.save(save_path_label,time_series_label)

    return time_series_data, time_series_label




if __name__=='__main__': 
    
    time_0 = time.time()
    data_path = "AnomalyDetection_UCR_classification\\*"

    data_paths = sorted(glob.glob(os.path.join(data_path, '*')))
    print('We have %s files in data_path.'%(len(data_paths)))

    class_set_not = []

    class_set = []
    for i_ in data_paths:
        i_ = os.path.normpath(i_)
        i_sep = i_.split(os.sep)
        if (i_sep[-2] not in class_set) and (i_sep[-2] not in class_set_not):
            class_set.append(i_sep[-2])
    
    # print(class_set)

    data_path_set = []
    for i_ in class_set:
        data_path_set_sub = []
        for j_ in data_paths:
            if i_ in j_ :
                data_path_set_sub.append(j_) 
        data_path_set.append(data_path_set_sub)
    # print(data_path_set)
    # quit()
    for path_sub in data_path_set:

        path_ = path_sub[0]
        # print(path_)
        # quit()
        df = pd.read_csv(path_, header=None).to_numpy()
        data = df[:, 1:].astype(float)
        label = df[:, 0]

        data_ls = []
        label_ls = []

        for line_idx_ in range(data.shape[0]):

            data_ = data[line_idx_]
            label_ = label[line_idx_]
            data_ls.append(data_)
            label_ls.append(int(label_-1))

        path_ = path_sub[1]
        print(path_)
        df = pd.read_csv(path_, header=None).to_numpy()

        data = df[:, 1:].astype(float)
        label = df[:, 0]


        for line_idx_ in range(data.shape[0]):

            data_ = data[line_idx_]
            label_ = label[line_idx_]
            data_ls.append(data_)
            label_ls.append(int(label_-1))

        path_ = os.path.normpath(path_)

        print('The length of origin dataset is %s'%(len(data_ls)))
        print('The shape of origin dataset is %s'%(data_ls[0].shape))

        N_class = np.max(label_ls) + 1
        n_iter_ = 5
        healthy_ratio_ = [0.0, 0.5]
        order_anomaly_ = [[-1,-2,-3],[-4,-5,-6],[-7,-8,-9],[-10,-11,-12]]
        class_choosen_ = list(range(N_class))

        # If there are too many categories
        if N_class > 5:
            continue

        for i_iter in range(n_iter_):
            # for i_healthy_ratio in healthy_ratio_:
            #    for i_order_anomaly in order_anomaly_:
            #       for i_class_choosen in class_choosen_:
            save_path_base = 'Data generated'
            save_path_class = path_.split(os.sep)[-2]
            save_path_root = os.path.join(save_path_base, save_path_class)
            try:
                os.makedirs(save_path_root)
            except:
                pass
            try:
                save_path_iter = 'Iter_%s'%(i_iter)
                time_series_data_, time_series_label_ = Parallel(n_jobs=16)(delayed(DataGenerator)(X=data_ls, y=label_ls, n_class=N_class, 
                                                                                                    healthy_ratio=i_healthy_ratio, class_choosen=i_class_choosen, 
                                                                                                    order_anomaly = i_order_anomaly, save_path = [save_path_root, save_path_iter]) 
                                                                                                    for i_healthy_ratio in healthy_ratio_
                                                                                                    for i_order_anomaly in order_anomaly_ 
                                                                                                    for i_class_choosen in class_choosen_)
                # time_series_data_, time_series_label_ = DataGenerator(X=data_ls, y=label_ls, n_class=N_class, healthy_ratio=i_healthy_ratio, class_choosen=i_class_choosen, order_anomaly = i_order_anomaly)

            except:
                continue
    print('Time used: %s'%(str(time.time()-time_0)))