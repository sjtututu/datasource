#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/8/13 14:30
# @Author: Tu Xinglong
# @File  : DataSource.py

# 内置模块
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import tushare as ts
import pandas as pd

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.abspath(os.path.dirname(CurrentPath))
sys.path += [Pre_path, Pre_path + '\\Engine']
from IndustryConst import (ZXFirstIndus, ZXSecondIndus, ZXThirdIndus)


"""中信一级行业成分股更新"""
def ZXClass(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\ZXClass\\"
    ZX_FirstIndusCode = ['0' + str(x) for x in range(15001, 15030, 1)]
    for induscode in ZX_FirstIndusCode:
        for date in trading_day:
            dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
            df = c.sector(induscode, dt)
            date_time = [
                datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0,
                         0) for i in range(int(len(df.Data) / 2))
            ]
            stock_code = [x[0:6] for x in df.Data[::2]]
            stock_name = df.Data[1::2]
            zx_class = pd.DataFrame([date_time, stock_code, stock_name],
                                    index=['EndDate', 'SecuCode',
                                           'SecuAbbr']).T
            zx_class.index = zx_class['EndDate'].tolist()
            if not os.path.isdir(data_path + induscode):
                os.mkdir(data_path + induscode)
                zx_class.to_hdf(data_path + induscode + '\\' + date + '.h5',
                                date)
            else:
                zx_class.to_hdf(data_path + induscode + '\\' + date + '.h5',
                                date)
    print("中信一级行业更新完毕！")


"""中信一、二、三级行业成分股更新"""
def ZXClass2(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\ZXClass2\\"
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        indus_data = []
        for induscode in ZXThirdIndus:
            zx_class = {}
            df = c.sector(induscode, dt)
            if df.ErrorCode != 10000009:
                zx_class["SecuCode"] = [x[0:6] for x in df.Data[::2]]
                zx_class["SecuAbbr"] = df.Data[1::2]
                zx_class["FirstIndusCode"] = [induscode[0:6]] * (int(
                    len(df.Data) / 2))
                zx_class["FirstIndusName"] = [ZXFirstIndus[induscode[0:6]]
                                              ] * (int(len(df.Data) / 2))
                zx_class["SecondIndusCode"] = [induscode[0:9]] * (int(
                    len(df.Data) / 2))
                zx_class["SecondIndusName"] = [ZXSecondIndus[induscode[0:9]]
                                               ] * (int(len(df.Data) / 2))
                zx_class["ThirdIndusCode"] = [induscode] * (int(
                    len(df.Data) / 2))
                zx_class["ThirdIndusName"] = [ZXThirdIndus[induscode]] * (int(
                    len(df.Data) / 2))
                zx_class = pd.DataFrame(zx_class,
                                        columns=[
                                            'SecuCode', 'SecuAbbr',
                                            'FirstIndusCode', 'FirstIndusName',
                                            'SecondIndusCode',
                                            'SecondIndusName',
                                            'ThirdIndusCode', 'ThirdIndusName'
                                        ])
            else:
                continue
            indus_data.append(zx_class)
        indus_data = pd.concat(indus_data).sort_values(by="SecuCode")
        indus_data.index = date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
        ] * indus_data.shape[0]
        indus_data.to_hdf(data_path + date + '.h5', date)
    print("中信行业信息更新完毕！")


"""全部A成分股更新"""
def Stock_A(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Stock_A_info\\"
    # 001004 全部A股
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = c.sector("001004", dt)
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(int(len(df.Data) / 2))
        ]
        stock_code = [x[0:6] for x in df.Data[::2]]
        stock_name = df.Data[1::2]
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
        df = c.sector("001023", dt)
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(int(len(df.Data) / 2))
        ]
        stock_code = [x[0:6] for x in df.Data[::2]]
        stock_name = df.Data[1::2]
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
        df = c.css(
            c.sector("001004", dt).Data[::2],
            "ISSURGEDLIMIT,ISDECLINELIMIT,NAME", "TradeDate=" + dt)
        data = pd.DataFrame(df.Data,
                            index=["LimitUp", "LimitDown", "SecuAbbr"]).T
        data = data[data.LimitUp.str.contains("是")
                    | data.LimitDown.str.contains("是")]
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


"""A股每日收益率更新"""
def Stock_Ret(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Stock_Ret\\"
    # 001023 全部风险警示股票
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        df = c.css(
            c.sector("001004", dt).Data[::2], "DIFFERRANGE", "TradeDate=" + dt)
        data = pd.DataFrame(df.Data, index=["Ret"]).T
        data["SecuCode"] = [x[0:6] for x in data.index.tolist()]
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(data.shape[0])
        ]
        data["EndDate"] = date_time
        data = data.reindex(columns=['EndDate', 'SecuCode', 'Ret'])
        data.index = data['EndDate'].tolist()
        data.to_hdf(data_path + date + '.h5', date)
    print("全市场股票收益率更新完毕！")


"""每日停复牌信息统计"""
def Stock_Suspend(trading_day):
    # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\Stock_Suspend\\"
    for date in trading_day:
        dt = date[0:4] + "-" + date[4:6] + "-" + date[6:8]
        # 截面行情接口
        df = c.css(
            c.sector("001004", dt).Data[::2], "TRADESTATUS", "TradeDate=" + dt)
        # 条件选股接口
        # df = c.cps("B_001004","TRADESTATUS,TRADESTATUS," + dt ,"CONTAINALL([TRADESTATUS],停牌) ",
        #            "orderby=ra([TRADESTATUS]),top=max([TRADESTATUS],10000),sectordate="+dt)
        data = pd.DataFrame(df.Data, index=["Status"]).T
        data["SecuCode"] = [x[0:6] for x in data.index.tolist()]
        date_time = [
            datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0, 0)
            for i in range(data.shape[0])
        ]
        data["EndDate"] = date_time
        data.index = data['EndDate'].tolist()
        data = data[data.Status.str.contains('连续停牌')
                    | data.Status.str.contains('停牌一天')]
        data = data.reindex(columns=['EndDate', 'SecuCode'])
        data.to_hdf(data_path + date + '.h5', date)
    print("每日停复牌股票更新完毕！")


"""中信行业哑变量因子更新"""
def Industry_ZX(trading_day):
    # # 设置文件路径
    data_path = "E:\\HistoryData\\MarketInfo\\Daily\\"
    ZX_FirstIndusCode = ['0' + str(x) for x in range(15001, 15030, 1)]
    for induscode in ZX_FirstIndusCode:
        for date in trading_day:
            stock_a = [
                x for x in pd.read_hdf(
                    data_path + "Stock_A_info\\" + date +
                    ".h5", date)["SecuCode"].tolist() if x[0:3] != "688"
            ]
            zx_data = pd.read_hdf(
                data_path + "ZXClass\\" + induscode + "\\" + date + ".h5",
                date)["SecuCode"].tolist()
            indux = [1 if x in zx_data else 0 for x in stock_a]
            zxindus = pd.DataFrame(indux, index=stock_a).T
            zxindus.index = [
                datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 0, 0,
                         0)
            ]
            if not os.path.isdir(data_path + "ZXIndus\\" + induscode):
                os.mkdir(data_path + "ZXIndus\\" +induscode)
                zxindus.to_hdf(data_path + "ZXIndus\\" + induscode + "\\" + date + ".h5",
                               date)
            else:
                zxindus.to_hdf(data_path + "ZXIndus\\" + induscode + "\\" + date + ".h5",
                               date)
    print("中信行业哑变量更新完毕！")


if __name__ == "__main__":
    """
    Choice数据源
    """
    from EmQuantAPI import *  # noqa
    c.start()  # noqa
    """设置下载日期"""
    pro = ts.pro_api()
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    start_date = str(TradeDate.iloc[0, 0])
    end_date = str(TradeDate.iloc[-1, 0])
    trading_day = [
        x for x in pro.trade_cal(exchange='SSE',
                                 start_date=start_date,
                                 end_date=end_date,
                                 is_open='1')['cal_date'].tolist()
    ]
    """行业更新"""
    ZXClass(trading_day)
    ZXClass2(trading_day)
    """股票更新"""
    Stock_A(trading_day)
    Stock_Limit(trading_day)
    Stock_ST(trading_day)
    Stock_Suspend(trading_day)
    Stock_Ret(trading_day)
    """哑变量更新"""
    Industry_ZX(trading_day)
