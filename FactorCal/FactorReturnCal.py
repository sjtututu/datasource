#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/6/26 22:00
# @Author: Tu Xinglong
# @File  : FactorReturnCal.py

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

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa

# 测试基准：2017-01-01去除风险警示后的所有上市股票
def singlefactor_return(params):
    '''
    单因子有效性检测
    :param params:
    :return:
    '''
    X_factorlist = params["X_factorlist"]
    Y_factorlist = params["Y_factorlist"]
    X_data = []
    stock_A_ref = pd.read_hdf("D:\\IntegPlatform\\AlphaData\\" + "stock_A_ref.h5", "stock_A_ref")
    stock_list = stock_A_ref["SecuCode"].tolist()
    for X_factor in X_factorlist:
        if X_factor[0:2] == 'ZX':
            factor_data = pd.read_hdf(FactorPath + "IndusFactor\\" + X_factor + ".h5", X_factor)
            factor_data = factor_data.reindex(columns=stock_list).fillna(0)
        else:
            factor_data = pd.read_hdf(FactorPath + "StyleFactor\\" + X_factor + ".h5", X_factor)
            factor_data = factor_data.reindex(columns=stock_list).fillna(0)
        X_data.append(factor_data.iloc[3390:5690,:])
        del factor_data
        gc.collect()
    Y_data = pd.read_hdf(FactorPath +"FinaFactor\\ret.h5", "ret").iloc[3390:5690,:].rolling(2).sum()
    Y_data = Y_data.reindex(columns=stock_list).fillna(0)
    for Y_factor in Y_factorlist:
        factor_data = pd.read_hdf(FactorPath +"ErrorFactor\\" + Y_factor + ".h5", Y_factor).iloc[3390:5690,:]
        factor_data = factor_data.reindex(columns=stock_list).fillna(0)
        X_data = X_data + [factor_data]
        X_para = np.zeros((len(X_factorlist)+1,factor_data.shape[1]))
        factor_ret = np.zeros((factor_data.shape[0],len(X_factorlist)+1))
        rows = factor_data.shape[0]
        for i in range(rows-2):
            Y_para = Y_data.iloc[i+2,:] # 周期T+2
            for j,data_x in enumerate(X_data):
                X_para[j,:] = data_x.iloc[i,:]
            model = OLS(x=X_para.T,y=np.array(Y_para))
            factor_ret[i,:] = model.beta
        factor_ret = pd.DataFrame(factor_ret, index=factor_data.index, columns=(X_factorlist+[Y_factor]))
        factor_ret["ret_cumsum"] = np.cumsum(factor_ret.iloc[:,38])
        factor_ret.to_csv(FactorPath+"ReturnFactor\\" +Y_factor + "_ret.csv" )
    del X_data,Y_data,factor_data,factor_ret
    gc.collect()
    return

# 所有因子残差收益率合并成DataFrame的形式
def factorerror_return(X_factorlist):
    '''
    有效因子进行回归，预测选股
    :return:
    '''
    stock_A_ref = pd.read_hdf("D:\\IntegPlatform\\AlphaData\\" + "stock_A_ref.h5", "stock_A_ref")
    stock_list = stock_A_ref["SecuCode"].tolist()
    # 这里选择需要进行股票预测的因子数据
    # alpha_errfactor = ['alpha002_err','alpha004_err','alpha005_err','alpha007_err','alpha008_err',
    #                    'alpha009_err','alpha011_err','alpha012_err','alpha013_err','alpha015_err',
    #                    'alpha016_err','alpha017_err','alpha018_err','alpha019_err','alpha021_err',
    #                    'alpha022_err','alpha024_err','alpha025_err','alpha026_err','alpha031_err',
    #                    'alpha032_err','alpha033_err','alpha034_err','alpha036_err','alpha037_err',
    #                    'alpha038_err','alpha039_err','alpha042_err','alpha044_err','alpha045_err',
    #                    'alpha050_err','alpha052_err','alpha054_err','alpha055_err','alpha057_err',
    #                    'alpha060_err','alpha062_err','alpha064_err','alpha072_err','alpha075_err',
    #                    'alpha078_err','alpha081_err','alpha083_err','alpha085_err','alpha088_err','alpha096_err']
    # tech_errfactor = ['tech001_err','tech002_err','tech005_err','tech006_err','tech008_err','tech012_err',
    #                   'tech015_err','tech016_err','tech023_err','tech025_err','tech026_err','tech032_err',
    #                   'tech034_err','tech037_err','tech039_err','tech047_err','tech054_err','tech061_err',
    #                   'tech062_err','tech073_err','tech083_err','tech090_err','tech092_err','tech099_err',
    #                   'tech104_err','tech111_err','tech113_err','tech114_err','tech115_err','tech125_err',
    #                   'tech127_err','tech130_err','tech140_err','tech142_err','tech163_err','tech166_err']
    # factorlist = X_factorlist + tech_errfactor
    factorlist = X_factorlist + [x[:-3] for x in os.listdir("D:\\HistoryData\\FactorData\\" + "ErrorFactor\\") if x[0:4]=="alph"]
    Y_data = pd.read_hdf(FactorPath + "FinaFactor\\ret.h5", "ret").iloc[3390:5690, :].rolling(2).sum()
    Y_data = Y_data.reindex(columns=stock_list).fillna(0)
    rows = Y_data.shape[0]
    factorerror_return = np.zeros((Y_data.shape[0], len(factorlist)))
    X_data = []
    for j, X_factor in enumerate(factorlist):
        if j < 29:
            factor_data = pd.read_hdf(FactorPath + "IndusFactor\\" + X_factor + ".h5", X_factor).iloc[3390:5690, :]
        elif j < 38:
            factor_data = pd.read_hdf(FactorPath + "StyleFactor\\" + X_factor + ".h5", X_factor).iloc[3390:5690, :]
        else:
            factor_data = pd.read_hdf(FactorPath + "ErrorFactor\\" + X_factor + ".h5", X_factor).iloc[3390:5690, :]
        factor_data = factor_data.reindex(columns=stock_list).fillna(0)
        X_data.append(factor_data)
        del factor_data
        gc.collect()
    for i in range(rows - 2):
        print(i)
        X_para = np.zeros((len(factorlist), Y_data.shape[1]))
        for j,data in enumerate(X_data):
            X_para[j, :] = data.iloc[i, :]
        Y_para = Y_data.iloc[i + 2, :]  # 周期T+2
        model = OLS(x=X_para.T, y=np.array(Y_para))
        factorerror_return[i, :] = model.beta
    factorerror_return = pd.DataFrame(factorerror_return, index=data.index, columns=factorlist)
    df = pd.DataFrame(np.zeros([3390,len(factorlist)]),columns=factorlist)
    factorerror_return = pd.concat([df,factorerror_return])
    factorerror_return.index = range(factorerror_return.shape[0])
    factorerror_return.to_hdf("D:\\IntegPlatform\\AlphaData\\"+ "factorerror_return.h5","factorerror_return",mode="w",format="fixed")
    return

def set_params(X_factorlist,Y_factorlist):
    td = {"X_factorlist": X_factorlist}
    params = []
    for i, factorlist in enumerate(Y_factorlist):
        td['Y_factorlist'] = factorlist
        params.append(td.copy())
    return params


if __name__=="__main__":
    indus_list    = [x[:-3] for x in os.listdir(FactorPath+"IndusFactor\\")]
    indus_list.sort(key=lambda x: int(re.sub("\D", "", x)))
    style_list    = [x[:-3] for x in os.listdir(FactorPath+"StyleFactor\\")]
    error_list    = [x[:-3] for x in os.listdir(FactorPath+"ErrorFactor\\")]
    X_factorlist  = indus_list + style_list 
    Y_factorlist  = error_list#,'techJLBL_err','techKPQK_err',,'techLFBL_err'['techYCCJL_err']#
    Y_factorlist = [Y_factorlist[i:i + 1] for i in range(0, len(Y_factorlist), 1)]
    paras = set_params(X_factorlist,Y_factorlist)
    pool = multiprocessing.Pool(5)
    pool.map(singlefactor_return, paras)
    pool.close()
    pool.join()
    # factorerror_return(X_factorlist)
    print()