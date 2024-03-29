import os
import csv
import json
import torch
import pandas as pd
from datetime import datetime
import numpy as np


def split_minute_data(minute_data, l, r, seq_len = 30, gap_len = 10, pre_len = 30, select_nums=5):
    daily_minutes = 240 
    X = []
    Y = []
    time_stamps = []
    for i in range(l, r, daily_minutes):
        start_index = i
        end_index = min(i + daily_minutes, r)
        for _ in range(select_nums):
            start = np.random.randint(start_index, end_index - seq_len - gap_len - pre_len + 1)
            x_l = start
            x_r = start + seq_len
            X.append(minute_data[:, x_l: x_r])
            y_l = start + seq_len + gap_len
            y_r = start + seq_len + gap_len + pre_len
            Y.append(minute_data[:, y_l: y_r])

            time_stamps.append((np.arange(x_l, x_r), np.arange(y_l, y_r)))
    return np.array(X), np.array(Y), np.array(time_stamps)

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,2), keepdims=True)
    std = train.std(axis=(0,1,2), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

np.random.seed(320427)
data_dir = '/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets'
data_name = 'CSI500'
date_format = r"%Y-%m-%d %H:%M:%S"
data_format = np.float32
with open(os.path.join(data_dir, data_name, "features_col.json"), 'r') as f:
    features_col = json.load(f)
with open(os.path.join(data_dir, data_name, "minute_features_col.json"), 'r') as f:
    minute_features_col = json.load(f)
with open(os.path.join(data_dir, data_name, "date.csv"), 'r') as f:
    reader = csv.reader(f)
    daily_date = [datetime.strptime(x[0], date_format) for x in reader] #[DT]
with open(os.path.join(data_dir, data_name, "minute_datetime.csv"), 'r') as f:
    reader = csv.reader(f)
    minute_datetime = [datetime.strptime(x[0], date_format) for x in reader] #[MT]

daily_features = torch.load(os.path.join(data_dir, data_name, "features.pth")).numpy().astype(data_format) #[N, DT, DF]
minute_features = torch.load(os.path.join(data_dir, data_name, "minute_features.pth")).numpy().astype(data_format) #[N, MT, MF]
reg_cap = pd.read_csv(os.path.join(data_dir, data_name, "reg_capital.csv")).values[:, 1].astype(data_format) #[N]
num_nodes = daily_features.shape[0] # = minute_features.shape[0] = 500

industry = daily_features[:, :, features_col.index('industry')] #[N, DT] == [N, repeat]
industry = industry[:, 0] #[N]
turnover_rate = daily_features[:, :, features_col.index('turnover_rate')] #[N, DT]
turnover_rate = np.mean(turnover_rate, axis=1).astype(data_format) + 1e-3 #[N]
minute_close = minute_features[:, :, minute_features_col.index('close')] #[N, MT] as label

reg_cap += 1e-3
industry_dist = reg_cap[np.newaxis, :] / reg_cap[:, np.newaxis] + turnover_rate[np.newaxis, :] / turnover_rate[:, np.newaxis]
industry_dist[industry[:, np.newaxis] != industry[np.newaxis, :]] = 0
# save daily industry distance as CSI500.npy
np.save(os.path.join(data_dir, data_name, "industry_dist.npy"), industry_dist)

minute_per_day = 240
num_days = int(minute_close.shape[1] / minute_per_day)
# split minute_close into train, val, test with 3:1:1
split_line1 = int(num_days * 0.6) * minute_per_day
split_line2 = int(num_days * 0.8) * minute_per_day
print("num_days:", num_days, split_line1, split_line2)

# calc minute_close pre mean for every stock
# for example: pre_mean[i,j] = mean(minute_close[i, :j])
# pre_mean = np.zeros_like(minute_close)
# for i in range(num_nodes):
#     pre_mean[i] = np.cumsum(minute_close[i], axis=0) / np.arange(1, minute_close.shape[1] + 1)
# # minute_close[i,j] /= pre_mean[i,j]
# minute_close = minute_close / (pre_mean + 1e-4) # to relative price



train_x, train_target, train_timestamp = split_minute_data(minute_close, 0, split_line1) #[B, N, T], [B, N, T], [B, 2, T]
val_x, val_target, val_timestamp  = split_minute_data(minute_close, split_line1, split_line2)
test_x, test_target, test_timestamp = split_minute_data(minute_close, split_line2, minute_close.shape[1])

# (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)


all_data = {
    'train': {
        'x': train_x,
        'target': train_target,
        'timestamp': train_timestamp,
    },
    'val': {
        'x': val_x,
        'target': val_target,
        'timestamp': val_timestamp,
    },
    'test': {
        'x': test_x,
        'target': test_target,
        'timestamp': test_timestamp,
    },
    # 'stats': {
    #     'pre_mean': pre_mean,
    # }
}

print(np.isnan(all_data['val']['x'], all_data['val']['target']).any())

print('train x:', all_data['train']['x'].shape)
print('train target:', all_data['train']['target'].shape)
print('train timestamp:', all_data['train']['timestamp'].shape)
print()
print('val x:', all_data['val']['x'].shape)
print('val target:', all_data['val']['target'].shape)
print('val timestamp:', all_data['val']['timestamp'].shape)
print()
print('test x:', all_data['test']['x'].shape)
print('test target:', all_data['test']['target'].shape)
print('test timestamp:', all_data['test']['timestamp'].shape)
print()
# print('train data _mean :', stats['_mean'].shape, stats['_mean'])
# print('train data _std :', stats['_std'].shape, stats['_std'])

np.savez_compressed(
    os.path.join(data_dir, data_name, "CSI500.npz"),
    train_x=all_data['train']['x'], train_target=all_data['train']['target'],
    train_timestamp=all_data['train']['timestamp'],
    valid_x=all_data['val']['x'], valid_target=all_data['val']['target'],
    valid_timestamp=all_data['val']['timestamp'],
    test_x=all_data['test']['x'], test_target=all_data['test']['target'],
    test_timestamp=all_data['test']['timestamp'],
    # pre_mean=all_data['stats']['pre_mean'],
    # mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
)