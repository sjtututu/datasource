#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/9 14:40
# @Author: Tu Xinglong
# @File  : SetUp.py

import os
import sys
import multiprocessing
import tushare as ts
import datetime as dt
import pandas as pd

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.abspath(os.path.dirname(CurrentPath))
sys.path += [CurrentPath, Pre_path + '\\Engine']
DataPath = Pre_path + '\\MarketData\\'


def set_params(file_list):
    new_file_list = []
    for file_name in file_list:
        new_file_list.append("python " + CurrentPath + "\\" + file_name)
    return new_file_list


if __name__ == "__main__":
    # """初始化tushare接口"""
    # ts.set_token("5206e26b853c22999a24bcb9be6c81824554e6ab7c6d55edc70b1844")
    pro = ts.pro_api()
    """判断当前日期是否已经更新"""
    stock_01 = pd.read_csv(DataPath + "Day\\SZE\\" + "000001.csv")
    last_date = str(stock_01['trade_date'].iloc[-1])  # 数据日期
    today_date = (dt.date.today()).strftime('%Y%m%d')  # 今天日期
    # 获取交易日
    Trading_Day = pro.trade_cal(exchange='SSE',
                                start_date=last_date,
                                end_date=today_date,
                                is_open=1)
    local_time = str(dt.datetime.now())[11:19]
    # 每天18点之后才开始更新
    if local_time >= "18:00:00":
        trading_day = [x for x in Trading_Day['cal_date'] if x not in last_date]
    else:
        trading_day = [x for x in Trading_Day['cal_date'] if (x not in last_date)
                       and (x not in today_date)]
    if trading_day == []:  # 如果为空则不更新
        print("The current_date is updated completely~")
    else:
        trading_day = pd.DataFrame(trading_day, index=range(len(trading_day)))
        trading_day.columns = ['trade_date']
        trading_day.to_csv(Pre_path + "\\TradeDate.csv",
                           index=False,
                           mode='w')
        # ===============================================================
        # ===============================================================
        """
        Tushare数据更新
        """
        file_list_1 = ['T_DataUpdate_Day.py','T_DataUpdate_Fina.py']
        new_file_list_1 = set_params(file_list_1)
        for pyfile in new_file_list_1:
            os.system(pyfile)
        print("All files update completely!")
