#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/6/25 12:40
# @Author: Tu Xinglong
# @File  : ErrorFactor.py

import os
import sys
import re
import gc
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pyfinance.ols import OLS, PandasRollingOLS
from sklearn import preprocessing
import tushare as ts

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa

def error_factor(params):
    length = params["length"]
    X_factorlist = params["X_factorlist"]
    Y_factorlist = params["Y_factorlist"]
    X_data = []
    for X_factor in X_factorlist:
        if X_factor[0:2] == 'ZX':
            factor_data = pd.read_hdf(FactorPath + "IndusFactor\\" + X_factor + ".h5", X_factor).iloc[-length:]
        else:
            factor_data = pd.read_hdf(FactorPath + "StyleFactor\\" + X_factor + ".h5", X_factor).iloc[-length:]
        X_data.append(factor_data)
        del factor_data
        gc.collect()
    for Y_factor in Y_factorlist:
        if Y_factor[0:5] == 'alpha':
            factor_data = pd.read_hdf(FactorPath +"Alpha101\\" + Y_factor + ".h5", Y_factor).iloc[-length:]
        elif Y_factor[0:5] == 'tech_':
            factor_data = pd.read_hdf(FactorPath + "TechFactor\\" + Y_factor + ".h5", Y_factor).iloc[-length:]
        else:
            factor_data = pd.read_hdf(FactorPath + "Tech191\\" + Y_factor + ".h5", Y_factor).iloc[-length:]
        err_data = np.ones(factor_data.shape) * np.nan
        X_para = np.zeros((len(X_factorlist),factor_data.shape[1]))
        tradingday = factor_data.index
        for i,date in enumerate(tradingday):
            Y_para = pd.DataFrame(factor_data.iloc[i,:])
            for j,data_x in enumerate(X_data):
                X_para[j,:] = data_x.iloc[i,:]
            model = OLS(x=X_para.T,y=np.array(Y_para))
            err_data[i,:] = model.resids
        err_data = pd.DataFrame(err_data, index=factor_data.index, columns=factor_data.columns)  # .iloc[-self.length:,:]
        err_data = err_data.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        err_data = remove_extreme(err_data, axis=1)  # 三倍标准差去极值
        err_data = pd.DataFrame(preprocessing.scale(err_data, axis=1), index=factor_data.index,
                            columns=factor_data.columns)  # 因子截面方向标准化，均值0方差1
        save_hdf(err_data,Y_factor + "_err",length)
    del X_factor,factor_data,err_data
    gc.collect()
    return

def set_params(X_factorlist,Y_factorlist,length):
    td = {"length" : length,
          "X_factorlist": X_factorlist}
    params = []
    for i, factorlist in enumerate(Y_factorlist):
        td['Y_factorlist'] = factorlist
        params.append(td.copy())
    return params


if __name__=="__main__":
    # 设置更新日期
    pro = ts.pro_api()
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    Start_Date = None
    End_Date = str(TradeDate.iloc[-1, 0])
    length = TradeDate.shape[0]  # 输出更新的数据长度
    indus_list    = [x[:-3] for x in os.listdir(FactorPath+"IndusFactor\\")]
    indus_list.sort(key=lambda x: int(re.sub("\D", "", x)))
    style_list    = [x[:-3] for x in os.listdir(FactorPath+"StyleFactor\\")]
    alpha101_list = [x[:-3] for x in os.listdir(FactorPath+"Alpha101\\")]
    tech191_list  = [x[:-3] for x in os.listdir(FactorPath+"Tech191\\")]
    tech_list     = [x[:-3] for x in os.listdir(FactorPath+"TechFactor\\")]
    X_factorlist  = indus_list + style_list
    Y_factorlist  = alpha101_list + tech191_list + tech_list
    # exit_list= [x[:-7] for x in os.listdir("D:\\HistoryData\\FactorData\\ErrorFactor\\")]
    # Y_factorlist = [x for x in Y_factorlist if x not in exit_list]
    Y_factorlist = [Y_factorlist[i:i + 1] for i in range(0, len(Y_factorlist), 1)]
    paras = set_params(X_factorlist,Y_factorlist,length)
    pool = multiprocessing.Pool(6)
    pool.map(error_factor, paras)
    pool.close()
    pool.join()
    print("The data of ErrorFactor update completely!")



