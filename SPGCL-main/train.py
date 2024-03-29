import argparse
import os
import torch
import numpy as np
import time
import torch.optim as optim
import lib.utils as utils
from lib.args import add_args
from torch.utils.data import DataLoader, Subset
from lib.dataloader import Slope, Traffic, Stock
from lib.layers.SPGCL import SPGCL
from Trainers import SPGCLTrainer

# Load and initialize other parameters
parser = argparse.ArgumentParser('StdModel')
add_args(parser, "CSI500")
args = parser.parse_args()
args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device)
args.save_dir = os.path.join('results', args.data, args.log_key)
args.fig_save_dir = os.path.join(args.save_dir, 'figs')
args.log_dir = os.path.join(args.save_dir, 'logs')
args.graph_dir = os.path.join(args.save_dir, 'graphs')
utils.makedirs(args.save_dir)
utils.makedirs(args.fig_save_dir)
utils.makedirs(args.graph_dir)
utils.set_random_seed(args.seed)

if __name__ == '__main__':
    # initialize
    model = SPGCL(args=args).to(args.device)

    loss = torch.nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # array for evaluation results
    normed_judge_list = []
    real_judge_list = []
    pems_result_list = []

    train_set = Stock(data_dir=args.data_dir, data_name=args.data, device=args.device, seq_len=args.ini_seq_len, gap_len=args.gap_len,
                        pre_len=args.pre_len, args=args, mode="train")
    test_set = Stock(data_dir=args.data_dir, data_name=args.data, device=args.device, seq_len=args.ini_seq_len, gap_len=args.gap_len,
                        pre_len=args.pre_len, args=args, mode="test")
    valid_set = Stock(data_dir=args.data_dir, data_name=args.data, device=args.device, seq_len=args.ini_seq_len, gap_len=args.gap_len,
                        pre_len=args.pre_len, args=args, mode="valid")
            

    scaler = train_set.scaler
    args.num_nodes = train_set.num_nodes

    # split test,train,valid set
    train_set = Subset(train_set, range(0, train_set.len))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_set = Subset(test_set, range(0, test_set.len))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    valid_set = Subset(valid_set, range(0, valid_set.len))
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)

    trainer = SPGCLTrainer(model, loss, optimizer, train_loader, valid_loader, test_loader, scaler, args)

    if args.mode == "train":
        # Warming up
        # trainer.train_embeddings()
        normed_judge, real_judge, pems_result = trainer.train(args=args)
    elif args.mode == "test":
        normed_judge, real_judge, pems_result = trainer.test(model, trainer.args, test_loader, scaler,
                                                             path=args.save_dir + r"/best_model.pth", mode=args.mode)
    else:
        raise ValueError

    normed_judge_list.append(normed_judge)
    real_judge_list.append(real_judge)
    pems_result_list.append(pems_result)

    for train_loop_idx in range(1, args.train_loop + 1):
        args.log_key = args.log_key + str(train_loop_idx)
        # train on subgraphs
        print("Change the train loop to {}".format(train_loop_idx))
        # every train loop ini slope set
        if "CSI500" in args.data:
            train_set = Stock(data_dir=args.data_dir, data_name=args.data, device=args.device, seq_len=args.ini_seq_len, gap_len=args.gap_len,
                                pre_len=args.pre_len, args=args, mode="train")

        scaler = train_set.scaler
        args.num_nodes = train_set.num_nodes

        train_set = Subset(train_set, range(0, train_set.len))
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

        trainer.train_loader = train_loader
        trainer.val_loader = valid_loader
        trainer.test_loader = test_loader
        trainer.args = args

        if args.mode == "train":
            normed_judge, real_judge, pems_result = trainer.train(args=args)
        elif args.mode == "test":
            normed_judge, real_judge, pems_result = trainer.test(model, trainer.args, test_loader, scaler,
                                                                 path=args.save_dir + r"/best_model.pth",
                                                                 mode=args.mode)
        else:
            raise ValueError

        normed_judge_list.append(normed_judge)
        real_judge_list.append(real_judge)
        pems_result_list.append(pems_result)

    # save test mean results
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    normed_judge_list = np.array(normed_judge_list)
    real_judge_list = np.array(real_judge_list)
    pems_result_list = np.array(pems_result_list)
    normed_judge, real_judge, pems_result = normed_judge_list.mean(axis=0), real_judge_list.mean(
        axis=0), pems_result_list.mean(axis=0)
    with open(args.save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(
            normed_judge[0], normed_judge[1], normed_judge[2], normed_judge[3], normed_judge[4], time_stamp,
            "normed_value", args.data, args.log_key, args.mode)
        fin.write(result)
    with open(args.save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(
            real_judge[0], real_judge[1], real_judge[2], real_judge[3], real_judge[4], time_stamp, "real_value",
            args.data, args.log_key, args.mode)
        fin.write(result)
    with open(args.save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(
            pems_result[0], pems_result[1], pems_result[2], pems_result[3], time_stamp, "real_value",
            args.data, args.log_key, args.mode)
        fin.write(result)
