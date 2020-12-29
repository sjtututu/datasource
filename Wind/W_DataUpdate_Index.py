#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/8/13 14:30
# @Author: Tu Xinglong
# @File  : DataSource.py

# 内置模块
import os
import sys
import warnings
import tushare as ts
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.abspath(os.path.dirname(CurrentPath))


"""沪深300指数权重更新"""
def HS300(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Weight\\HS300\\"
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = w.wset("indexconstituent", "date=" + dt + ";windcode=000300.SH")
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(len(df.Data[0]))
        ]
        stock_code = [x[0:6] for x in df.Data[1]]
        stock_name = df.Data[2]
        stock_weight = df.Data[4]
        hs300_data = pd.DataFrame(
            [date_time, stock_code, stock_name, stock_weight],
            index=['EndDate', 'SecuCode', 'SecuAbbr', 'WeightRatio']).T
        hs300_data.index = hs300_data['EndDate'].tolist()
        hs300_data.to_hdf(data_path + date + ".h5", date)
    print("沪深300权重更新完毕！")


"""中证500指数权重更新"""
def ZZ500(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Weight\\ZZ500\\"
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = w.wset("indexconstituent", "date=" + dt + ";windcode=000905.SH")
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(len(df.Data[0]))
        ]
        stock_code = [x[0:6] for x in df.Data[1]]
        stock_name = df.Data[2]
        stock_weight = df.Data[4]
        zz500_data = pd.DataFrame(
            [date_time, stock_code, stock_name, stock_weight],
            index=['EndDate', 'SecuCode', 'SecuAbbr', 'WeightRatio']).T
        zz500_data.index = zz500_data['EndDate'].tolist()
        zz500_data.to_hdf(data_path + date + ".h5", date)
    print("中证500权重更新完毕！")


"""SZ50,HS300,ZZ500,ZZ800,ZZ1000收益率更新"""
def Index_Ret(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Index_Ret\\"
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = w.wsd("000016.SH,000300.SH,000905.SH,000906.SH,000852.SH",
                   "pct_chg", dt, dt, "PriceAdj=F")
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(len(df.Data[0]))
        ]
        index_code = [x[0:6] for x in df.Codes]
        index_name = ["上证50", "沪深300", "中证500", "中证800", "中证1000"]
        ret = df.Data[0]
        index_ret = pd.DataFrame(
            [date_time, index_code, index_name, ret],
            index=['EndDate', 'IndexCode', 'IndexAbbr', 'Ret']).T
        index_ret.index = index_ret['EndDate'].tolist()
        index_ret.to_hdf(data_path + date + ".h5", date)
    print("指数收益率更新完毕！")

"""全部A成分股更新"""
def Stock_A(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Stock_A_info\\"
    # 001004 全部A股
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = w.wset("sectorconstituent","date="+dt+";sectorid=a001010100000000")
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(len(df.Data[0]))
        ]
        stock_code = [x[0:6] for x in df.Data[1]]
        stock_name = df.Data[2]
        stock_a = pd.DataFrame([date_time, stock_code, stock_name],
                               index=['EndDate', 'SecuCode', 'SecuAbbr']).T
        stock_a.index = stock_a['EndDate'].tolist()
        stock_a.to_hdf(data_path + date + '.h5', date)
    print("全A成分股更新完毕！")

"""A股风险警示股票更新"""
def Stock_ST(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Stock_ST\\"
    # 001023 全部风险警示股票
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = w.wset("sectorconstituent","date="+dt+";sectorid=1000006526000000")
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(len(df.Data[0]))
        ]
        stock_code = [x[0:6] for x in df.Data[1]]
        stock_name = df.Data[2]
        stock_st = pd.DataFrame([date_time, stock_code, stock_name],
                                index=['EndDate', 'SecuCode', 'SecuAbbr']).T
        stock_st.index = stock_st['EndDate'].tolist()
        stock_st = stock_st[~stock_st.SecuAbbr.str.contains('B')]
        stock_st.to_hdf(data_path + date + '.h5', date)
    print("风险警示股票更新完毕！")

"""A股每日涨跌停股票统计"""
def Stock_Limit(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Stock_Limit\\"
    # 001023 全部风险警示股票
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        stock_a = w.wset("sectorconstituent","date="+dt+";sectorid=a001010100000000")
        df = w.wss(stock_a.Data[1],"maxupordown","tradeDate="+dt)
        data = pd.DataFrame([stock_a.Data[1], stock_a.Data[2], df.Data[0]],
                            index=["SecuCode", "SecuAbbr", "Limit"]).T#["LimitUp", "LimitDown", "SecuAbbr"]
        # data.to_csv("./limit.csv")

        data_up = data[data.Limit.str.contains(1)]

        data["LimitUp"] = data["Limit"].replace([1],["是"])
        data["LimitDown"] = data["Limit"].replace([-1],["否"])
        data = data[data.LimitUp.str.contains("是")
                    | data.LimitDown.str.contains("否")]
        data["SecuCode"] = [x[0:6] for x in data.index.tolist()]
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(data.shape[0])
        ]
        data["EndDate"] = date_time
        data = data.reindex(columns=[
            'EndDate', 'SecuCode', 'SecuAbbr', "LimitUp", "LimitDown"
        ])
        data.index = data['EndDate'].tolist()
        data.to_hdf(data_path + date + '.h5', date)
    print("每日涨跌停股票更新完毕！")

if __name__ == "__main__":
    """
    Wind数据源
    """
    from WindPy import *
    w.start()
    """设置下载日期"""
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    start_date = str(TradeDate.iloc[0, 0])
    end_date = str(TradeDate.iloc[-1, 0])
    pro = ts.pro_api()
    trading_day = [
        x for x in pro.trade_cal(exchange='SSE',
                                 start_date=start_date,
                                 end_date=end_date,
                                 is_open='1')['cal_date'].tolist()
    ]
    # """指数更新"""
    HS300(trading_day)
    ZZ500(trading_day)
    Index_Ret(trading_day)
    Stock_Limit(trading_day)
