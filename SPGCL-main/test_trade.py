from lib.trader import Strategy, Trader
from lib.dataloader import  Stock
import numpy as np
import torch
from lib.trade_model import TraderModel
test_model_name = "SPGCL"

minute_data = torch.load("/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets/CSI500/minute_features.pth")
print(minute_data.size())
close_data = minute_data[:, :, 1].numpy()
model = TraderModel(test_model_name)
strategy = Strategy(model)

trader = Trader(close_data, strategy)
trader.run()