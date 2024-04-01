import numpy as np
import pandas as pd

import torch
import numpy as np
import time
class Action:
    def __init__(self, stock, type, price, amount, timestamp):
        self.stock = stock
        self.type = type
        self.price = price
        self.amount = amount
        self.timestamp = timestamp

class Strategy:
    day_minutes = 240
    transaction_num = 2
    
    def __init__(self, model, cash = 1e5, seq_len = 30, gap_len = 10, pre_len = 30):
        self.model = model
        self.history_data = []
        self.actions = []
        self.stocks = {}
        self.seq_len = seq_len
        self.gap_len = gap_len
        self.pre_len = pre_len
        self.day = 0
        self.cash = cash
        self.fees_rate = 5e-3
        self.brokerage_rate = 3e-4
        self.trade_minute_bias = np.random.randint(0, 10)
        self.todo_actions = {}
    def check_decrease(self, stock):
        return self.history_data[-1][stock] < self.history_data[-2][stock]
    def update_data(self, data):
        self.history_data.append(data)
        timestamp = (self.day, len(self.history_data))
        if len(self.history_data) == self.day_minutes:
            for i in range(500):
                if i in self.stocks and self.stocks[i].type == "buy":
                    self.actions.append(Action(i, "sell", data[i], self.stocks[i].amount, timestamp))
            self.history_data = []
            self.stocks = {}
            self.day += 1
            self.trade_minute_bias = np.random.randint(0, 10)
            self.todo_actions = {}
        elif len(self.history_data) > self.trade_minute_bias:

            if (len(self.history_data) -  self.trade_minute_bias) % (self.seq_len + self.gap_len) == self.seq_len and len(self.history_data) + self.gap_len + self.pre_len < self.day_minutes:
                # time to predict next
                return_rates = self.model.predict(self.history_data[-self.seq_len:])
                print("return_rates shape is ", return_rates.shape)
                print(return_rates.max())
                return_rates = return_rates.reshape(-1)
                # select max 10 stocks by return rate
                top_stocks = np.argsort(return_rates)[-10:]
                # print(top_stocks)
                for stock_id in top_stocks:
                    print(return_rates[stock_id])
                    if return_rates[stock_id] < (self.fees_rate * 2 + self.brokerage_rate) or data[stock_id] < 1e-3:
                        continue
                    # print("buy stock: ", stock_id, "price: ", data[stock_id], "amount: ", self.cash // 20 // data[stock_id], "timestamp: ", timestamp)
                    action = Action(stock_id, "buy", data[stock_id], self.cash // 20 // data[stock_id], timestamp)
                    buy_time = len(self.history_data) + self.gap_len
                    if buy_time in self.todo_actions:
                        self.todo_actions[buy_time].append(action)
                    else:
                        self.todo_actions[buy_time] = [action]
            
            # check every minute and stock not hold over 30 mins
            for i in range(500):
                if i in self.stocks:
                    if (self.check_decrease(i) and data[i] > (1 + self.fees_rate * 2 + self.brokerage_rate) * self.stocks[i].price) or (timestamp[1] == self.stocks[i].timestamp[1] + self.pre_len):
                        self.actions.append(Action(i, "sell", data[i], self.stocks[i].amount, timestamp))
            self.actions = self.actions + self.todo_actions.get(len(self.history_data), [])
    def get_actions(self):
        return self.actions
    def update_status(self, cash, actions):
        self.cash = cash
        self.actions = []
        for action in actions:
            if action.type == "buy":
                self.stocks[action.stock] = action
            else:
                del self.stocks[action.stock]
        pass
    

class Trader:
    def __init__(self, data, strategy, cash = 1e5, fees_rate = 5e-3, brokerage_rate = 3e-4, start_minute_time = 0):
        self.data = data #[days, stocks, minutes]
        self.strategy = strategy
        self.strategy.update_status(cash, [])
        self.stocks = np.zeros(data.shape[1])
        self.fees_rate = fees_rate
        self.fees = 0
        self.cash = cash
        self.brokerage_rate = brokerage_rate
        self.brokerage = 0
        self.start_minute_time = start_minute_time
    def run(self):
        for minute in range(2400):
            self.strategy.update_data(self.data[:, minute])
            actions = self.strategy.get_actions()
            ret_actions = []
            for action in actions:
                turnover = self.data[action.stock, minute] * action.amount
                # print(action.type)
                if action.type == "buy" and self.cash >= turnover * (1 + self.fees_rate + self.brokerage_rate):
                    self.cash -= turnover * (1 + self.fees_rate + self.brokerage_rate)
                    self.fees += turnover * self.fees_rate
                    self.stocks[action.stock] += action.amount
                    self.brokerage += turnover * self.brokerage_rate
                    ret_actions.append(action)
                    print("buy stock: ", action.stock, "price: ", self.data[action.stock, minute], "amount: ", action.amount, "timestamp: ", action.timestamp, "cash: ", self.cash)
                elif action.type == "sell" and self.stocks[action.stock] >= action.amount:
                    self.cash += turnover * (1 - self.fees_rate)
                    self.stocks[action.stock] -= action.amount
                    self.fees += turnover * self.fees_rate
                    print("sell stock: ", action.stock, "price: ", self.data[action.stock, minute], "amount: ", action.amount, "timestamp: ", action.timestamp, "cash: ", self.cash)
            self.strategy.update_status(self.cash, ret_actions)
            if minute % 240 == 0:
                print("day: {}, cash: {}".format(self.strategy.day, self.cash))
                print("check stock sum:", self.stocks.sum())


