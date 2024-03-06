import os
import sys
import csv
import json
import torch
sys.path.append('/home/wbyan/stock_data')

from data_utils import load_data #会显示报错，但是上面加了sys path可以使用

stock_list_name="csi500"
stock_save_path = "/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets/{}".format(stock_list_name.upper())
if not os.path.exists(stock_save_path):
    os.mkdir(stock_save_path)
#读取股市数据

date_list, features, labels, stock_list = load_data(
        data_type="ch_daily_processed",
        stock_list=stock_list_name,
        start_date="2022-01-01",
        end_date="2023-12-31",
        contain_bj=False,
        drop_short=False,
        short_len=200,
)
torch.save(features, stock_save_path+"/features.pth")
with open(stock_save_path+"/date.csv", 'w') as f:
    writer = csv.writer(f)
    for timestamp in date_list:
        writer.writerow([timestamp])
date_list, features, labels, stock_list, data = load_data(
        data_type="ch_minute_origin",
        stock_list=stock_list_name,
        start_date="2022-01-01",
        end_date="2023-12-31",
        contain_bj=False,
        drop_short=False,
        short_len=200,
)
torch.save(features, stock_save_path+"/minute_features.pth")
with open(stock_save_path+"/minute_datetime.csv", 'w') as f:
    writer = csv.writer(f)
    for timestamp in date_list:
        writer.writerow([timestamp])

reg_capital = load_data(
        data_type='reg_captital',
        stock_list=stock_list,
    )
reg_capital.to_csv(stock_save_path+"/reg_capital.csv")

minute_features_col = ['open','close','low','high','volume','money','factor','high_limit','low_limit','avg','pre_close','paused']
with open(stock_save_path+"/minute_features_col.json", 'w') as f:
    json.dump(minute_features_col, f)
features_col = [
    "open",
    "pct_chg",
    "high",
    "low",
    "close",
    "pre_close",
    "ma5",
    "ma10",
    "ma15",
    "ma20",
    "ma25",
    "change",
    "vol",
    "amount",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "total_mv",
    "circ_mv",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount",
    "up_limit",
    "down_limit",
    "industry",
]
with open(stock_save_path+"/features_col.json", 'w') as f:
    json.dump(features_col, f)

'''
    data_type 读取的数据类型 暂时可读取的有  
        "ch_daily_processed" a股日频数据（处理后）
         'ch_daily_origin' a股日频数据（原始数据）

         'reg_captital' 读取股票注册信息
         'feature_col_index' 读取字段在feature中的列index
         'label_col_index'读取字段在label中的列index


    col_name: 仅读取  'feature_col_index'和 'label_col_index'有效，获取某字段的index
             
    stock_list：股票列表 "all"全部  "csi500"中证500 "csi300"沪深300 "csi800"沪深300+中证500，或自定义list ['000001.SZ','000002.SZ']
    start_date="2010-01-01"起始时间
    end_date="2022-12-31",结束时间
    contain_bj=False,是否包含北交所股票
    drop_short=False,是否去掉交易日过少的股票
    short_len=200, 如果drop_short=True ，则去掉交易日少于short_len的股票

    return:
        若 读取"ch_daily_processed"或'ch_daily_origin' 
            date_list ：区间交易日列表
            features : 特征 包含feature_col
            
            labels : 标签 包含   label_col 

            stock_list  ：所读取的股票列表
若去除交易天数不符股票和北交所股票，可能读取csi800/500/300/all并不是全部股票
'''
#feature字段名称可以从tushare上查询
"""
feature_col = [
        "open",
        "pct_chg",
        "high",
        "low",
        "close",
        "pre_close",
        "ma5",
        "ma10",
        "ma15",
        "ma20",
        "ma25",
        "change",
        "vol",
        "amount",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "buy_sm_vol",
        "buy_sm_amount",
        "sell_sm_vol",
        "sell_sm_amount",
        "buy_md_vol",
        "buy_md_amount",
        "sell_md_vol",
        "sell_md_amount",
        "buy_lg_vol",
        "buy_lg_amount",
        "sell_lg_vol",
        "sell_lg_amount",
        "buy_elg_vol",
        "buy_elg_amount",
        "sell_elg_vol",
        "sell_elg_amount",
        "net_mf_vol",
        "net_mf_amount",
        "up_limit",
        "down_limit",
        "industry",
    ]

label_col = ["return_10", 未来10天收益率均值
    "return_5", 未来5天收益率均值
    "validity_label", 样本有效性(由于对齐问题进行数据存在补全，标记该条记录是实际数据还是补全数据)
    "pct_chg"   未来一日的收益率
    ]
"""