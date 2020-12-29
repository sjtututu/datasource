#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/5 21:55
# @Author: Tu Xinglong
# @File  : IndusFactor.py

import os
import sys

import pandas as pd

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.abspath(os.path.dirname(CurrentPath))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
DataPath = Pre_path + '\\MarketData\\'
InfoPath = Pre_path + '\\MarketInfo\\'
FactorPath = Pre_path + '\\FactorData\\'

from LoggingPlus import Logger  # noqa
import MySQLConn  # noqa


def indus_data(ZX_Class, TradingDay):
    FirstIndusNumber = ZX_Class['FirstIndusNumber']
    SecuCode = [x[0:6] for x in ZX_Class['SecuCode']]
    TradingDay = TradingDay['cal_date'].drop_duplicates()
    for class_number in range(1, 30, 1):
        stock_code = [
            SecuCode[i][0:6] for i, number in enumerate(FirstIndusNumber)
            if number == class_number
        ]
        df_class = pd.DataFrame([1] * len(stock_code), index=stock_code).T
        df_class = df_class.reindex(
            columns=sorted(SecuCode), index=range(TradingDay.shape[0])).fillna(
            method='ffill').fillna(0).astype('int32')
        df_class.index = TradingDay
        df_class.to_hdf(FactorPath + "\\IndusFactor\\" + "ZX_Class_" +
                        str(class_number) + ".h5",
                        "ZX_Class_" + str(class_number),
                        mode='w')
    return


if __name__ == "__main__":
    ZX_Class = pd.read_csv(InfoPath + "\\StockInfo\\"
                           "BasicData\\ZX_Class.csv",
                           encoding="GBK")
    TradingDay = pd.read_csv(InfoPath + "\\StockInfo\\"
                             "BasicData\\Trading_Day.csv")
    indus_data(ZX_Class, TradingDay)
    print("The data of IndusFactor update completely!")