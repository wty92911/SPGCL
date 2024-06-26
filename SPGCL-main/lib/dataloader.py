import math
import random

import os
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from lib.normalization import *
from numpy.lib.stride_tricks import sliding_window_view
from dtaidistance import dtw
from sklearn.neighbors import NearestNeighbors


def rotation(array, point, mode, angle):
    def Nrotation_angle_get_coor_coordinates(point, center, angle):
        # print(point)
        src_x, src_y = point[:, 0], point[:, 1]
        center_x, center_y = center
        radian = math.radians(angle)

        dest_x = ((src_x - center_x) * np.cos(radian) + (src_y - center_y) * np.sin(radian) + center_x)
        dest_y = ((src_y - center_y) * np.cos(radian) - (src_x - center_x) * np.sin(radian) + center_y)

        return dest_x, dest_y

    def Srotation_angle_get_coor_coordinates(point, center, angle):
        src_x, src_y = point[:, 0], point[:, 1]
        center_x, center_y = center
        radian = math.radians(angle)

        dest_x = ((src_x - center_x) * np.cos(radian) - (src_y - center_y) * np.sin(radian) + center_x)
        dest_y = ((src_x - center_x) * np.sin(radian) + (src_y - center_y) * np.cos(radian) + center_y)

        return [dest_x, dest_y]

    array_2D = np.array(array[:, 0:2])  # choose 2D coordinate
    array_high = np.array(array[:, 2])

    if mode == "nrotation":
        array_rot_x, array_rot_y = Nrotation_angle_get_coor_coordinates(array_2D, point[0:2], angle)
    else:
        array_rot_x, array_rot_y = Srotation_angle_get_coor_coordinates(array_2D, point[0:2], angle)
    return np.concatenate([array_rot_x.reshape(len(array_rot_x), 1), array_rot_y.reshape(len(array_rot_x), 1),
                           array_high.reshape(len(array_rot_x), 1)], axis=1)


class Slope(Dataset):
    def __init__(self, data_dir, mode="Train", data_name="HZY_east", seq_len=3, pre_len=1, device='cuda:0', relative=True, now_step=0, args=None, train_set=True, prepare_data=False):
        self.device = device
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.format = "float32"
        self.features = None

        locs_path = os.path.join(data_dir, data_name,  "{}_locs_{}_{}.npy".format(data_name, seq_len, pre_len))
        data_path = os.path.join(data_dir, data_name, "{}.csv".format(data_name))
        # Read, select, and re-formating.
        data = np.loadtxt(fname=data_path, skiprows=1, delimiter=",").astype(self.format)
        self.total_num_nodes = data.shape[0]
        print("Shape of {}: {} ".format(data_path, data.shape), ", select shape: {} nodes {}/{} features".format(self.total_num_nodes, args.feature_volume, data.shape[1]-3))

        if prepare_data == True:
            self.features, _ = self.normalize_dataset(data[:, 3:], "std", False)

        # Masking subgraph
        mask, self.edge_index, self.edge_index_negative = self.masking(loc=data[:, :3], args=args)
        data = data[mask]
        self.num_nodes = data.shape[0]
        # train on subgraph
        if os.path.exists(locs_path):           # load loc
            self.locs = np.load(locs_path)[mask].astype(self.format)
        else:
            self.locs, self.loc_scaler = self.normalize_dataset(data[:, :3], "std", True)

        features, self.scaler = self.normalize_dataset(data[:, 3:], "std", False)  # [N, T]
        train_test_margin = int(data.shape[1] * args.train)
        # select and normalize features
        if train_set:
            start_ = np.random.randint(low=3, high=train_test_margin - args.feature_volume, size=1)[0]
            features = features[:, start_: start_+args.feature_volume]
        else:
            features = features[:, train_test_margin+1:]
        # Store some information. len is decided by self.get_X_and_Y.
        self.len = features.shape[-1] - self.seq_len - self.pre_len + 1
        abs_X, abs_Y = self.get_X_and_Y(features)  # [B, N, seq_len], [B, N, pre_len]

        if not relative:
            self.trainX, self.trainY = abs_X, abs_Y
        else:
            # N*N. ab_dist, h_dist，dtw, temp_features
            ab_dist = cdist(self.locs, self.locs)
            h_dist = np.tile(self.locs[:, 2:3], self.num_nodes) - np.transpose(self.locs[:, 2:3])
            dtw_save_path = os.path.join(r"./datasets", data_name, "{}_dtw.npy".format(data_name))
            if not os.path.exists(dtw_save_path):
                dtw_list = dtw.distance_matrix(self.features, parallel=False, show_progress=True)
                np.save(dtw_save_path, dtw_list)
            # 2D array masking, which is the same as dtw_dist[mask,:][:,mask]
            dtw_dist = np.load(dtw_save_path).astype(self.format)[np.ix_(mask, mask)]

            # [len, N, N, 1]
            ab_dist = ab_dist[np.newaxis, :, :, np.newaxis].repeat(self.len, axis=0).astype(self.format)
            h_dist = h_dist[np.newaxis, :, :, np.newaxis].repeat(self.len, axis=0).astype(self.format)
            dtw_dist = dtw_dist[np.newaxis, :, :, np.newaxis].repeat(self.len, axis=0).astype(self.format)
            # [N, F] -> [N, N, F]
            temp_feature = features[:, np.newaxis, :].repeat(self.num_nodes, axis=1).astype(self.format)
            # [N, N, relative features]
            relative_feature = temp_feature - temp_feature.transpose(1, 0, 2)
            # [len, N, N, seq_len/pre_len]
            X, Y = self.get_X_and_Y(relative_feature)
            W = np.concatenate((X, ab_dist, h_dist, dtw_dist), axis=3)

            self.trainX = torch.from_numpy(W).float()
            self.trainY = torch.from_numpy(abs_Y.copy()).float()
        self.seq_len = self.trainX.shape[-1]
        # Standardize the characteristics of all neighbors of each point so that each point selects its neighbors
        # by its own criteria, rather than different points with the same criteria
        self.trainX = (self.trainX - self.trainX.mean(dim=2, keepdim=True)) / self.trainX.std(dim=2, keepdim=True)
        self.trainX = self.trainX.to(self.device)  # [B, N, N, seq_len + f_r]
        self.trainX_abs = torch.tensor(abs_X).to(self.device)
        self.trainY = self.trainY.to(self.device)  # [B, N, N, pre_len + f_r]

    def __getitem__(self, index):
        # every time return [1, N, seq_len]
        x = self.trainX[index]
        xx = self.trainX_abs[index]
        # [1, N, pre_len]
        y = self.trainY[index]
        return x, xx, y

    def __len__(self):
        return self.len

    def masking(self, loc, args):
        # subgraph is made by mask on adj matrix. return mask and edge_index
        if args.data_volume > self.total_num_nodes:
            args.data_volume = self.total_num_nodes
        mask = np.sort(np.random.choice(a=self.total_num_nodes, size=args.data_volume, replace=False))
        # rotation
        loc = rotation(loc, loc[random.randint(1, len(loc)-1)], "nrotation", random.random()*90)
        # positive graph ini
        A = self.load_init_adj(loc=loc, args=args)  # load from file, A is the positive graph
        A = A[np.ix_(mask, mask)]
        # negative graph ini
        B = self.load_init_adj_negative(loc=loc, args=args)  # load from file, A is the positive graph
        B = B[np.ix_(mask, mask)]
        return mask, torch.from_numpy(A).to_sparse().to(self.device), torch.from_numpy(B).to_sparse().to(self.device)

    def load_init_adj(self, loc, args):
        # Load initial adj
        path = os.path.join(args.data_dir, args.data, "{}_KNN_adj.npy".format(args.data))
        # if file exists:
        # if os.path.exists(path):
        #     A = np.load(path)
        if False:
            A = np.load(path)
        else:
            # Find the nearest k points
            nearestneighbors_euclidean = NearestNeighbors(
                n_neighbors=args.positives + 1, metric="euclidean").fit(loc)
            A = nearestneighbors_euclidean.kneighbors_graph(loc).toarray().T
            # if j is i point neighbors : j->i A[j,i]=1.
            np.save(path, A)
        return A

    def load_init_adj_negative(self, loc, args):
        path = os.path.join(args.data_dir, args.data, "{}_KFF_adj.npy".format(args.data))
        # if file exists:
        # if os.path.exists(path):
        #     A = np.load(path)
        if False:
            A = np.load(path)
        else:
            # Find the farthest k points
            nearestneighbors_euclidean = NearestNeighbors(
                n_neighbors=loc.shape[0], metric="euclidean").fit(loc)
            A = nearestneighbors_euclidean.kneighbors_graph(loc, mode="distance").toarray()
            B = A.copy()
            # find a border
            B.sort()
            boarder = B[:, -args.negative_init]
            for (i, item) in enumerate(A):
                A[i] = np.where(A[i] >= boarder[i], np.ones_like(A[i]), np.zeros_like(A[i]))
            A = A.T
            np.save(path, A)
        return A

    @staticmethod
    def max_min_norm(data):
        max_value = np.max(data)
        min_value = np.min(data)
        return (data - min_value)/(max_value - min_value)

    @staticmethod
    def normalize_dataset(data, normalizer, column_wise=False):
        # shape=[N, features]. If column_wise==True, the features will be normalized respectively.
        if normalizer == 'max01':
            if column_wise:
                minimum = data.min(axis=0, keepdims=True)
                maximum = data.max(axis=0, keepdims=True)
            else:
                minimum = data.min()
                maximum = data.max()
            scaler = MinMax01Scaler(minimum, maximum)
            data = scaler.transform(data)
            print('Normalize the dataset by MinMax01 Normalization')
        elif normalizer == 'max11':
            if column_wise:
                minimum = data.min(axis=0, keepdims=True)
                maximum = data.max(axis=0, keepdims=True)
            else:
                minimum = data.min()
                maximum = data.max()
            scaler = MinMax11Scaler(minimum, maximum)
            data = scaler.transform(data)
            print('Normalize the dataset by MinMax11 Normalization')
        elif normalizer == 'std':
            if column_wise:
                mean = data.mean(axis=0, keepdims=True)
                std = data.std(axis=0, keepdims=True)
            else:
                mean = data.mean()
                std = data.std()
            scaler = StandardScaler(mean, std)
            data = scaler.transform(data)
            print('Normalize the dataset by Standard Normalization')
        elif normalizer == 'None':
            scaler = NScaler()
            data = scaler.transform(data)
            print('Does not normalize the dataset')
        elif normalizer == 'cmax':
            # column min max, to be depressed
            # note: axis must be the spatial dimension, please check !
            scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
            data = scaler.transform(data)
            print('Normalize the dataset by Column Min-Max Normalization')
        else:
            raise ValueError
        return data, scaler

    def get_X_and_Y(self, data):
        # [N, T] or [N, N ,T] matrix.  split according yo seq and pre length at the last dim
        temp = sliding_window_view(data, window_shape=self.seq_len + self.pre_len, axis=-1)
        X, Y = np.split(temp, [self.seq_len], axis=-1)

        if X.ndim == 4:
            X = X.transpose(2, 0, 1, 3)
            Y = Y.transpose(2, 0, 1, 3)
        elif X.ndim == 3:
            X = X.transpose(1, 0, 2)
            Y = Y.transpose(1, 0, 2)
        return X, Y


class Traffic(Dataset):
    def __init__(self, data_dir, mode="Train", data_name="PEMS03", seq_len=3, pre_len=1, relative=True, now_step=0, args=None, train_set=True, prepare_data=False):
        self.device = args.device
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.format = "float32"
        self.dataname = data_name
        data_path = os.path.join(data_dir, data_name, "{}.npz".format(data_name))

        # Read, select, and re-formating.
        data_raw = np.load(data_path)
        data = (data_raw.f.data.transpose(2, 1, 0))[0]
        self.total_num_nodes = data.shape[0]

        if prepare_data == True:
            self.features, _ = self.normalize_dataset(data, "std", False)

        print("Shape of {}: {} ".format(data_path, data.shape), ", select shape: {} nodes {}/{} features".format(self.total_num_nodes, args.feature_volume, data.shape[1]))

        # Masking subgraph
        mask, self.edge_index, self.edge_index_negative = self.masking(loc=data[:, 1:], args=args)
        self.mask = mask
        data = data[mask]
        self.num_nodes = data.shape[0]

        # select and normalize features
        features, self.scaler = self.normalize_dataset(data, "std", False)  # [N, T]
        train_test_margin = int(data.shape[1] * args.train)
        # select and normalize features
        if train_set:
            start_ = np.random.randint(low=3, high=train_test_margin - args.feature_volume, size=1)[0]
            features = features[:, start_: start_+args.feature_volume]
        else:
            start_ = np.random.randint(low=train_test_margin+1, high=data.shape[1], size=1)[0]
            features = features[:, start_: start_ + args.feature_volume]

        # Store some information. len is decided by self.get_X_and_Y.
        self.len = features.shape[-1] - self.seq_len - self.pre_len + 1
        abs_X, abs_Y = self.get_X_and_Y(features)  # [B, N, seq_len], [B, N, pre_len]


        if not relative:
            self.trainX, self.trainY = abs_X, abs_Y
        else:
            temp_feature = features[:, np.newaxis, :].repeat(self.num_nodes, axis=1).astype(self.format)
            # [N, N, relative features]
            relative_feature = temp_feature - temp_feature.transpose(1, 0, 2)
            # [len, N, N, seq_len/pre_len]
            X, Y = self.get_X_and_Y(relative_feature)
            W = X

            self.trainX = torch.from_numpy(W).float()
            self.trainY = torch.from_numpy(abs_Y.copy()).float()

        self.seq_len = self.trainX.shape[-1]
        self.trainX = (self.trainX - self.trainX.mean(dim=2, keepdim=True)) / self.trainX.std(dim=2, keepdim=True)
        self.trainX = self.trainX.to(self.device)  # [B, N, N, seq_len + f_r]
        self.trainX_abs = torch.tensor(abs_X).to(self.device)
        self.trainY = self.trainY.to(self.device)  # [B, N, pre_len]

    def __getitem__(self, index):
        x = self.trainX
        data_name = self.dataname
        dtw_list_path_i = os.path.join(r"./datasets", data_name, "{}_dtw_{}.npy".format(data_name, index+1))
        dtw_dist = np.load(dtw_list_path_i).astype(self.format)[np.ix_(self.mask, self.mask)]
        dtw_dist = dtw_dist[np.newaxis, :, :, np.newaxis].repeat(self.len, axis=0).astype(self.format)

        x = torch.concat((x, torch.tensor(dtw_dist).cuda(device=self.device)), axis=3)
        x = x[index]
        xx = self.trainX_abs[index]
        y = self.trainY[index]
        self.seq_len = x.shape[-1]
        return x, xx, y

    def __len__(self):
        return self.len

    def masking(self, loc, args):
        if args.data_volume > self.total_num_nodes:
            args.data_volume = self.total_num_nodes
        mask = np.sort(np.random.choice(a=self.total_num_nodes, size=args.data_volume, replace=False))
        A = self.load_init_adj(loc=loc, args=args)  # load from file, A is the positive graph
        A = A[np.ix_(mask, mask)]
        # neg
        B = self.load_init_adj_negative(loc=loc, args=args)  # load from file, A is the positive graph
        B = B[np.ix_(mask, mask)]
        return mask, torch.from_numpy(A).to_sparse().to(self.device), torch.from_numpy(B).to_sparse().to(self.device)

    def load_init_adj(self, loc, args):
        # Load initial adj
        path = os.path.join(args.data_dir, args.data, "{}_KNN_adj.npy".format(args.data))
        # if file exists:
        if os.path.exists(path):
            A = np.load(path)
        else:
            # Find the nearest k points
            nearestneighbors_euclidean = NearestNeighbors(
                n_neighbors=args.positives + 1, metric="euclidean").fit(loc)
            A = nearestneighbors_euclidean.kneighbors_graph(loc).toarray().T
            np.save(path, A)
        return A

    def load_init_adj_negative(self, loc, args):
        path = os.path.join(args.data_dir, args.data, "{}_KFF_adj.npy".format(args.data))
        # if file exists:
        if os.path.exists(path):
            A = np.load(path)
        else:
            # Find the farest k points
            nearestneighbors_euclidean = NearestNeighbors(
                n_neighbors=loc.shape[0], metric="euclidean").fit(loc)
            A = nearestneighbors_euclidean.kneighbors_graph(loc, mode="distance").toarray()
            B = A.copy()
            B.sort()
            boarder = B[:, -args.negative_init]
            for (i, item) in enumerate(A):
                A[i] = np.where(A[i] >= boarder[i], np.ones_like(A[i]), np.zeros_like(A[i]))
            A = A.T
            np.save(path, A)
        return A

    @staticmethod
    def max_min_norm(data):
        max_value = np.max(data)
        min_value = np.min(data)
        return (data - min_value)/(max_value - min_value)

    @staticmethod
    def normalize_dataset(data, normalizer, column_wise=False):
        # shape=[N, features]. If column_wise==True, the features will be normalized respectively.
        if normalizer == 'max01':
            if column_wise:
                minimum = data.min(axis=0, keepdims=True)
                maximum = data.max(axis=0, keepdims=True)
            else:
                minimum = data.min()
                maximum = data.max()
            scaler = MinMax01Scaler(minimum, maximum)
            data = scaler.transform(data)
            print('Normalize the dataset by MinMax01 Normalization')
        elif normalizer == 'max11':
            if column_wise:
                minimum = data.min(axis=0, keepdims=True)
                maximum = data.max(axis=0, keepdims=True)
            else:
                minimum = data.min()
                maximum = data.max()
            scaler = MinMax11Scaler(minimum, maximum)
            data = scaler.transform(data)
            print('Normalize the dataset by MinMax11 Normalization')
        elif normalizer == 'std':
            if column_wise:
                mean = data.mean(axis=0, keepdims=True)
                std = data.std(axis=0, keepdims=True)
            else:
                mean = data.mean()
                std = data.std()
            scaler = StandardScaler(mean, std)
            data = scaler.transform(data)
            print('Normalize the dataset by Standard Normalization')
        elif normalizer == 'None':
            scaler = NScaler()
            data = scaler.transform(data)
            print('Does not normalize the dataset')
        elif normalizer == 'cmax':
            # column min max, to be depressed
            # note: axis must be the spatial dimension, please check !
            scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
            data = scaler.transform(data)
            print('Normalize the dataset by Column Min-Max Normalization')
        else:
            raise ValueError
        return data, scaler

    def get_X_and_Y(self, data):
        temp = sliding_window_view(data, window_shape=self.seq_len + self.pre_len, axis=-1)
        X, Y = np.split(temp, [self.seq_len], axis=-1)
        if X.ndim == 4:
            X = X.transpose(2, 0, 1, 3)
            Y = Y.transpose(2, 0, 1, 3)
        elif X.ndim == 3:
            X = X.transpose(1, 0, 2)
            Y = Y.transpose(1, 0, 2)
        return X, Y
import csv
import json
import torch
import tqdm
import pandas as pd
from datetime import datetime

class Stock(Dataset):
    def __init__(self, data_dir, mode="train", data_name="CSI500", seq_len=30, gap_len = 10, pre_len=30, device='cuda:0', args=None, prepare_data=False):
        np.random.seed(320427)
        self.device = device
        self.mode = mode
        self.seq_len = seq_len
        self.gap_len = gap_len
        self.pre_len = pre_len
        self.format = "float32"
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = args.batch_size
        self.date_format = r"%Y-%m-%d %H:%M:%S"
        self.cur_batch_index = -1

        file_data = np.load(r"/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets/CSI500/CSI500.npz")
        
        train_x = file_data[mode + '_x']  
        self.train_x = torch.from_numpy(train_x).float().to(device)
        train_target = file_data[mode + '_target'] 
        train_target = torch.from_numpy(train_target).float().to(device) #[B, N, T]
        # transform price to yield
        self.train_target = (train_target[:, :, -1] - train_target[:, :, 0]) / (train_target[:, :, 0] + 1e-5) # expected yield

        timestamp = file_data[mode + '_timestamp'] # index in minute data
        self.x_timestamp = timestamp[:, 0, :]
        self.y_timestamp = timestamp[:, 1, :]
        self.industry_dist = torch.from_numpy(np.load(r"/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets/CSI500/industry_dist.npy")).float().to(device)
        
        with open(os.path.join(data_dir, data_name, "features_col.json"), 'r') as f:
            features_col = json.load(f)
        
        daily_features = torch.load(os.path.join(data_dir, data_name, "features.pth")).numpy().astype(self.format) #[N, DT, DF]
        # reg_cap = pd.read_csv(os.path.join(data_dir, data_name, "reg_capital.csv")).values[:, 1].astype(self.format) #[N]
        self.total_num_nodes = daily_features.shape[0] # = minute_features.shape[0] = 500
        self.num_nodes = 500

        # print("Shape of {}: {} ".format(data_name, daily_features.shape), ", select shape: {} nodes {}/{} features".format(self.total_num_nodes, args.feature_volume, daily_features.shape[1]))
        
        # # daily data
        pct_chg = daily_features[:, :, features_col.index('pct_chg')] #[N, DT]
        # turnover_rate = daily_features[:, :, features_col.index('turnover_rate')] #[N, DT]
        # total_mv = daily_features[:, :, features_col.index('total_mv')] #[N, DT]
        # pe = daily_features[:, :, features_col.index('pe')] #[N, DT]
        
        
        
        # Masking subgraph
        self.train_x, self.scaler = self.normalize_dataset(self.train_x, "None", False)
        self.len = self.train_x.shape[0]
        mask, self.edge_index, self.edge_index_negative = self.masking(loc=pct_chg[:, -480:], args=args) #TODO: find a better feature as loc
        self.mask = mask
        # pct_chg = pct_chg[mask, :] # mask 之后 num nodes 可能变化
        # turnover_rate = turnover_rate[mask, :]
        # total_mv = total_mv[mask, :]
        # pe = pe[mask, :]
        # reg_cap = reg_cap[mask]


        # self.len = train_x.shape[0]
        # self.trainX_abs = torch.tensor(abs_X)
        # print("self.len = {}, self.seq_len = {}".format(self.len, self.seq_len))
        
        # # [N, N, DT]    
        # daily_mv_dist = total_mv[np.newaxis, :, :] - total_mv[:, np.newaxis, :]
        # daily_pe_dist = pe[np.newaxis, :, :] - pe[:, np.newaxis, :]
        # daily_industry_dist = reg_cap[np.newaxis, :, np.newaxis] - reg_cap[:, np.newaxis, np.newaxis] + turnover_rate[np.newaxis, :, :] - turnover_rate[:, np.newaxis, :]
        # daily_industry_dist[industry[:, np.newaxis, :] != industry[np.newaxis, :, :]] = 0
        # relative_daily_pct_chg = pct_chg[:, np.newaxis, :] - pct_chg[np.newaxis, :, :]

        # relative_X = abs_X[:, np.newaxis, :, :] - abs_X[:, :, np.newaxis, :]
        # means = []
        # print("relative_X shape{}".format(relative_X.shape))
        # for batch_start in range(0, relative_X.shape[0], self.batch_size):
        #     W = []
        #     for i in range(batch_start, min(batch_start + self.batch_size, relative_X.shape[0])):
        #         w = np.concatenate((relative_X[i], daily_mv_dist[:, :, daily_index[i], np.newaxis], daily_pe_dist[:, :, daily_index[i], np.newaxis], 
        #                         daily_industry_dist[:, :, daily_index[i], np.newaxis], relative_daily_pct_chg[:, :, daily_index[i], np.newaxis]), axis=2)
        #         W.append(w)
        #     print("Concatente i = {}".format(batch_start))
        #     tensor_w = torch.from_numpy(np.stack(W, axis=0)).float()
        #     means.append(tensor_w.mean(dim=1, keepdim=True))
        #     torch.save(tensor_w, os.path.join(data_dir, data_name, "temp_tensor/{}X_{}.pth").format(mode, batch_start))
        #     W = []

        # print("Concatenated Done")
        # mean = torch.cat(means, dim=0).mean(dim=0, keepdim=True)
        # std_sum = 0
        # for batch_start in range(0, relative_X.shape[0], self.batch_size):
        #     print("Calc mean i = {}".format(batch_start))
        #     tensor_w = torch.load(os.path.join(data_dir, data_name, "temp_tensor/{}X_{}.pth").format(mode, batch_start))
        #     std_sum += (tensor_w - mean).pow(2).sum()
        
        # std = (std_sum / (relative_X.shape[0] * relative_X.shape[1] * relative_X.shape[2] * relative_X.shape[3] - 1)).sqrt()
        # print("mean = {}, std = {}".format(mean, std))
        # for batch_start in range(0, relative_X.shape[0], self.batch_size):
        #     print("Normalize i = {}".format(batch_start))
        #     tensor_w = torch.load(os.path.join(data_dir, data_name, "temp_tensor/{}X_{}.pth").format(mode, batch_start))
        #     tensor_w = (tensor_w - mean) / (std + 1e-6)
        #     tensor_w = torch.save(tensor_w, os.path.join(data_dir, data_name, "temp_tensor/{}X_{}.pth").format(mode, batch_start))
        
        

    def __getitem__(self, index):
        x = self.industry_dist
        xx = self.train_x[index]
        y = self.train_target[index]
        
        return x, xx, y

    def __len__(self):
        return self.len
    
    def masking(self, loc, args):
        if args.data_volume > self.num_nodes:
            args.data_volume = self.num_nodes
        mask = np.sort(np.random.choice(a=self.num_nodes, size=args.data_volume, replace=False))
        A = self.load_init_adj(loc=loc, args=args)  # load from file, A is the positive graph
        A = A[np.ix_(mask, mask)]
        # neg
        B = self.load_init_adj_negative(loc=loc, args=args)  # load from file, A is the positive graph
        B = B[np.ix_(mask, mask)]
        return mask, torch.from_numpy(A).to_sparse().to(self.device), torch.from_numpy(B).to_sparse().to(self.device)

    def load_init_adj(self, loc, args):
        # Load initial adj
        path = os.path.join(args.data_dir, args.data, "{}_KNN_adj.npy".format(args.data))
        # if file exists:
        if os.path.exists(path):
            A = np.load(path)
        else:
            # Find the nearest k points
            nearestneighbors_euclidean = NearestNeighbors(
                n_neighbors=args.positives + 1, metric="euclidean").fit(loc)
            A = nearestneighbors_euclidean.kneighbors_graph(loc).toarray().T
            np.save(path, A)
        return A

    def load_init_adj_negative(self, loc, args):
        path = os.path.join(args.data_dir, args.data, "{}_KFF_adj.npy".format(args.data))
        # if file exists:
        if os.path.exists(path):
            A = np.load(path)
        else:
            # Find the farest k points
            nearestneighbors_euclidean = NearestNeighbors(
                n_neighbors=loc.shape[0], metric="euclidean").fit(loc)
            A = nearestneighbors_euclidean.kneighbors_graph(loc, mode="distance").toarray()
            B = A.copy()
            B.sort()
            boarder = B[:, -args.negative_init]
            for (i, item) in enumerate(A):
                A[i] = np.where(A[i] >= boarder[i], np.ones_like(A[i]), np.zeros_like(A[i]))
            A = A.T
            np.save(path, A)
        return A
    def get_X_and_Y(self, data):
        # [N, T] or [N, N ,T] matrix.  split according yo seq and pre length at the last dim
        temp = sliding_window_view(data, window_shape=self.seq_len + self.gap_len + self.pre_len, axis=-1)
        X, Y = np.split(temp, [self.seq_len], axis=-1)
        Y = Y[:, self.gap_len:]
        if X.ndim == 4:
            X = X.transpose(2, 0, 1, 3)
            Y = Y.transpose(2, 0, 1, 3)
        elif X.ndim == 3:
            X = X.transpose(1, 0, 2)
            Y = Y.transpose(1, 0, 2)
        return X, Y
    @staticmethod
    def normalize_dataset(data, normalizer, column_wise=False):
        # shape=[N, features]. If column_wise==True, the features will be normalized respectively.
        if normalizer == 'max01':
            if column_wise:
                minimum = data.min(axis=0, keepdims=True)
                maximum = data.max(axis=0, keepdims=True)
            else:
                minimum = data.min()
                maximum = data.max()
            scaler = MinMax01Scaler(minimum, maximum)
            data = scaler.transform(data)
            print('Normalize the dataset by MinMax01 Normalization')
        elif normalizer == 'max11':
            if column_wise:
                minimum = data.min(axis=0, keepdims=True)
                maximum = data.max(axis=0, keepdims=True)
            else:
                minimum = data.min()
                maximum = data.max()
            scaler = MinMax11Scaler(minimum, maximum)
            data = scaler.transform(data)
            print('Normalize the dataset by MinMax11 Normalization')
        elif normalizer == 'std':
            if column_wise:
                mean = data.mean(axis=0, keepdims=True)
                std = data.std(axis=0, keepdims=True)
            else:
                mean = data.mean()
                std = data.std()
            scaler = StandardScaler(mean, std)
            data = scaler.transform(data)
            print('Normalize the dataset by Standard Normalization')
        elif normalizer == 'None':
            scaler = NScaler()
            data = scaler.transform(data)
            print('Does not normalize the dataset')
        elif normalizer == 'cmax':
            # column min max, to be depressed
            # note: axis must be the spatial dimension, please check !
            scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
            data = scaler.transform(data)
            print('Normalize the dataset by Column Min-Max Normalization')
        else:
            raise ValueError
        return data, scaler