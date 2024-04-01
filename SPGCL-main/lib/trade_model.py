import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import lib.utils as utils
from lib.args import add_args
from torch.utils.data import DataLoader, Subset
from lib.dataloader import Slope, Traffic, Stock
from lib.layers.SPGCL import SPGCL
from Trainers import SPGCLTrainer

class TraderModel:
    def __init__(self, name):
        self.name = name
        if name == "SPGCL":
            parser = argparse.ArgumentParser('StdModel')
            add_args(parser, "CSI500")
            self.args = parser.parse_args()
            self.args.device = torch.device('cuda:' + str("0") if torch.cuda.is_available() else 'cpu')
            # self.args.device = torch.device('cpu')
            # torch.cuda.set_device(self.args.device)

            print(self.args.device)
            self.industry_dist = torch.from_numpy(np.load("/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets/CSI500/industry_dist.npy")).to(self.args.device)
            self.industry_dist = self.industry_dist.reshape(*self.industry_dist.shape, 1)
            self.graph = torch.load("/home/undergrad2023/tywang/SPGCL/SPGCL-main/results/CSI500/graph.pth")
            self.model = SPGCL(args=self.args).to(self.args.device)
            self.graph.to(self.args.device)
            model_path = "/home/undergrad2023/tywang/SPGCL/SPGCL-main/results/CSI500/training/best_model.pth"
            check_point = torch.load(model_path)
            state_dict = check_point['state_dict']
            self.model.load_state_dict(state_dict)
            self.model.to(self.args.device)
    def predict(self, data):
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
        elif type(data) == list:
            data = torch.tensor(data)
        data = data.to(self.args.device)
        if data.shape[0] != 500:
            data = data.T
        if self.name == 'SPGCL':
            return self.model.get_prediction([data, self.industry_dist, self.graph]).cpu().detach().numpy()