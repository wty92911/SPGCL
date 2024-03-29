# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import json
import os
import warnings
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from lib.dataloader import Stock
from lib.layers import LSTM
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


def load_data():
    file_data = np.load('/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets/CSI500/CSI500_sampled.npz')
    train_x = file_data['train_x']  
    train_target = file_data['train_target'] 
    train_timestamp = file_data['train_timestamp']

    val_x = file_data['val_x']
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    pre_mean = file_data['pre_mean']
    return train_x, train_target, train_timestamp, val_x, val_target, val_timestamp, test_x, test_target, test_timestamp, pre_mean
def re_pre_normalization(x, time_stamps, pre_mean):
    '''
    :param x: np.ndarray(b, N, t)
    :param time_stamps: np.ndarray, (b, t)
    :param pre_mean: np.ndarray, (N, T)
    :return:
    '''

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j, :] = x[i, j, :] * pre_mean[j, time_stamps[i]]
    return x


# mode = "real data"
mode = "normalize data"
data_dir = r"./datasets"
data_name = r"CSI500"
num_nodes = 500


seq_len = 12
gap_len = 16
pre_len = 12


train_x, train_y, train_timestamp, val_x, val_y, val_timestamp, test_x, test_y, test_timestamp, pre_mean = load_data()

method = 'LSTM'  #### SVR or ARIMA or LSTM


def evaluation(labels, predicts, T, pre_len, acc_threshold=0.05):
    """
    evalution the labels with predicts
    rmse, Root Mean Squared Error
    mae, mean_absolute_error
    F_norm, Frobenius norm
    Args:
        labels:
        predicts:
        acc_threshold: if lower than this threshold we regard it as accurate
    Returns:
    """
    if mode == "normalize data":
        labels = re_pre_normalization(labels, T, pre_mean)
        predicts = re_pre_normalization(predicts, T, pre_mean)
    labels = labels.reshape([-1, pre_len])
    predicts = predicts.reshape([-1, pre_len])
    # labels = labels.squeeze(0).astype("float32")
    # predicts = predicts.squeeze(0).astype("float32")
    rmse = mean_squared_error(y_true=labels, y_pred=predicts, squared=True)
    # mae = mean_squared_error(y_true=labels, y_pred=predicts, squared=False)
    mae = mean_absolute_error(y_true=np.array(labels), y_pred=np.array(predicts))
    r2 = r2_score(y_true=labels, y_pred=predicts)
    evs = explained_variance_score(y_true=labels, y_pred=predicts)
    # acc = a[np.abs(a - b) < np.abs(a * acc_threshold)]
    a, b = labels, predicts
    acc = a[np.abs(a - b) < np.abs(acc_threshold)]
    acc = np.size(acc) / np.size(a)
    # mape = MAPE(a, b)
    return rmse, mae, acc, r2, evs

############ SVR #############

if method == 'SVR':
    svr_model = [SVR(kernel='rbf', C=1e3, gamma=0.1) for _ in range(pre_len)]

    
    for i in range(train_x.shape[0]):
        print("fitting step: ", i, "/", train_x.shape[0])
        X = train_x[i]
        y = train_y[i]
        X = X.reshape(-1, seq_len)
        y = y.reshape(-1, pre_len)
        for j in range(pre_len):
            svr_model[j].fit(X, y[:, j])
    Y = []
    T = []
    for i in range(test_x.shape[0]):
        print("predicting step: ", i, "/", test_x.shape[0])
        X = test_x[i]
        X = X.reshape(-1, seq_len)
        y = []
        T.append(test_timestamp[i, 1])
        for j in range(pre_len):
            y.append(svr_model[j].predict(X))
        y = np.array(y).T
        Y.append(y)
    Y = np.stack(Y, axis=0)

    rmse, mae, accuracy, r2, var = evaluation(test_y, Y, T, pre_len, acc_threshold=0.1)
    print('SVR_rmse:%r' % rmse,
          'SVR_mae:%r' % mae,
          'SVR_acc:%r' % accuracy,
          'SVR_r2:%r' % r2,
          'SVR_var:%r' % var)
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = rmse, mae, accuracy, r2, var

######## ARIMA #########
elif method == 'ARIMA':
    warnings.filterwarnings("ignore")
    Y = []
    T = []
    for i in range(test_x.shape[0]):
        print("fitting step: ", i, "/", test_x.shape[0])
        X = test_x[i]
        X = X.reshape(-1, seq_len)
        y = []
        T.append(test_timestamp[i, 1])
        for k in range(num_nodes):
            model = ARIMA(X[k], order=(1, 1, 0))
            model_fit = model.fit()
            y.append(model_fit.forecast(steps=pre_len))
        y = np.array(y)
        Y.append(y)
    Y = np.stack(Y, axis=0)
    rmse, mae, accuracy, r2, var = evaluation(test_y, Y, T, pre_len, acc_threshold=0.1)
    print('ARIMA_rmse:%r' % rmse,
          'ARIMA_mae:%r' % mae,
          'ARIMA_acc:%r' % accuracy,
          'ARIMA_r2:%r' % r2,
          'ARIMA_var:%r' % var)
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = rmse, mae, accuracy, r2, var
elif method == 'LSTM':
    model = LSTM.train(train_x, train_y, num_nodes, torch.device("cuda:3"))
    Y = LSTM.predict(model, test_x, torch.device("cuda:3"))
    rmse, mae, accuracy, r2, var = evaluation(test_y, Y, test_timestamp[:, 1], pre_len, acc_threshold=0.1)
    print('LSTM_rmse:%r' % rmse,
          'LSTM_mae:%r' % mae,
          'LSTM_acc:%r' % accuracy,
          'LSTM_r2:%r' % r2,
          'LSTM_var:%r' % var)
    rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = rmse, mae, accuracy, r2, var

result_type = method
hyper_parameters = 'None'
time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

pure_data_file = "./results/pure_result.csv"
with open(pure_data_file, mode='a') as fin:
    str = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(rmse_ts, mae_ts, acc_ts, r2_ts, var_ts,
                                                                       time_stamp, hyper_parameters, data_name,
                                                                       result_type, mode)
    fin.write(str)
