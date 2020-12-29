#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/9 8:28
# @Author: Tu Xinglong
# @File  : C_DataUpdate_Fina.py

import os
import sys
import time
from itertools import chain

import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime
import warnings
from EmQuantAPI import *  # noqa

c.start()  # noqa

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
DataPath = Pre_path + '\\MarketData\\'
InfoPath = Pre_path + '\\MarketInfo\\'
FactorPath = Pre_path + '\\FactorData\\'

from LoggingPlus import Logger  # noqa


class Data_Update(object):
    '''
    Choice财务数据接口:
    用于下载股票财务报表，
    量化因子数据等
    '''

    def __init__(self, startdate, enddate, log):
        '''
        定义起始日期、数据库接口
        tushare接口初始化
        '''
        self.Start_Date = startdate
        self.End_Date = enddate
        self.log = log

    def save_h5(sefl, factor_data, factor_name):
        factor_list = ["sestni_FY1", "sestni_FY3", "eps_FY1"]  # 增量更新的因子
        if factor_name in factor_list:
            data_old = pd.read_hdf(
                FactorPath + "FinaFactor\\" + factor_name + ".h5", factor_name)
            factor_data = pd.concat([data_old, factor_data], axis=0)
            factor_data = factor_data.fillna(method="ffill").fillna(0)
            # factor_data=factor_data.drop(["trade_date"],axis=1)
            factor_data.to_hdf(FactorPath + "FinaFactor\\" + factor_name +
                               ".h5", factor_name, mode="w", format="fixed")
        else:
            factor_data = factor_data.fillna(method="ffill").fillna(0)
            factor_data.to_hdf(FactorPath + "FinaFactor\\" + factor_name +
                               ".h5", factor_name, mode="w", format="fixed")
        return

    def finance_data(self):
        Start_Date = self.Start_Date
        End_Data = self.End_Date
        log = self.log
        retry_count = 100
        pause = 1
        '''
        基本面数据
        财务数据
        技术指标
        :return:
        '''
        # 获取交易日
        data = c.tradedates("19960101", End_Date,
                            "period=1,order=1,market=CNSESH")
        tradingday = pd.DataFrame([str(parse(x))[0:4] + str(parse(x))[5:7] + str(
            parse(x))[8:10] for x in data.Data], columns=["trade_date"])
        # 001004 所有股票代码，每半年更新一次股票
        data = c.sector("001004", End_Date)
        stock_list = [stock_code for i, stock_code in enumerate(data.Data) if i % 2 == 0]
        report_date = ['1996-12-31', '1997-12-31', '1998-12-31', '1999-12-31', '2000-12-31',
                       '2001-12-31', '2002-12-31', '2003-12-31', '2004-12-31', '2005-12-31',
                       '2006-12-31', '2007-12-31', '2008-12-31', '2009-12-31', '2010-12-31',
                       '2011-12-31', '2012-12-31', '2013-12-31', '2014-12-31', '2015-12-31',
                       '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31']
        """定期年报实际披露日期、归属母公司股东净利润"""
        actul_date = pd.DataFrame(
            np.ones((len(report_date), len(stock_list))) * np.nan)
        net_profit = pd.DataFrame(
            np.ones((len(report_date), len(stock_list))) * np.nan)
        for i, re_date in enumerate(report_date):
            for _ in range(retry_count):
                try:
                    print(re_date)
                    data = c.css(stock_list, "STMTACTDATE",
                                 "ReportDate=" + re_date + ",Ispandas=1")
                    actul_date.iloc[i, :] = [str(parse(x))[0:4] + str(parse(x))[5:7] + str(
                        parse(x))[8:10] if x is not None else np.nan for x in data["STMTACTDATE"].values]
                    data = c.css(stock_list, "IncomeStatement", "ReportDate=" +
                                 re_date + ",type=1,ItemsCode=61,Ispandas=1")
                    net_profit.iloc[i, :] = data["INCOMESTATEMENT"].values
                    break
                except Exception:
                    time.sleep(pause)
        sestni_FY0 = pd.DataFrame()  # 归属母公司股东净利润增长率
        for i, stock_code in enumerate(stock_list):
            for _ in range(retry_count):
                try:
                    df = pd.DataFrame(
                        [actul_date.iloc[:, i].values, net_profit.iloc[:, i].values]).T
                    df.columns = ["trade_date", stock_code]
                    df = df.drop_duplicates(['trade_date'], keep='last')
                    df = df.dropna(axis=0)
                    df = pd.merge(df, tradingday, on="trade_date",
                                  how="outer", sort=True)
                    df = df.fillna(method="ffill").fillna(0)
                    df = pd.merge(df, tradingday, on="trade_date",
                                  how="inner", sort=True)
                    sestni_FY0[stock_code] = df[stock_code]
                    break
                except Exception:
                    time.sleep(pause)
        sestni_FY0.index = tradingday['trade_date']
        sestni_FY0.columns = [x[0:6] for x in stock_list]
        self.save_h5(sestni_FY0, "sestni_FY0")
        """过去五年营业总收入复合增长率、归属母公司净利润复合增长率"""
        year_list = ['1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
                     '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
                     '2016', '2017', '2018', '2019']
        cagrpni = pd.DataFrame(
            np.ones((len(year_list), len(stock_list))) * np.nan)
        cagrgr = pd.DataFrame(
            np.ones((len(year_list), len(stock_list))) * np.nan)
        for i, year in enumerate(year_list):
            for _ in range(retry_count):
                try:
                    data = c.css(stock_list, "CAGRPNI,CAGRGR",
                                 "Year=" + year + ",N=5,Ispandas=1")
                    cagrpni.iloc[i, :] = data['CAGRPNI'].values
                    cagrgr.iloc[i, :] = data["CAGRGR"].values
                    break
                except Exception:
                    time.sleep(pause)
        cagrpni_PY5 = pd.DataFrame()  # 过去五年营业总收入复合增长率
        cagrgr_PY5 = pd.DataFrame()  # 归属母公司净利润复合增长率
        for i, stock_code in enumerate(stock_list):
            try:
                df1 = pd.DataFrame(
                    [actul_date.iloc[:, i].values, cagrpni.iloc[:, i].values]).T
                df2 = pd.DataFrame(
                    [actul_date.iloc[:, i].values, cagrgr.iloc[:, i].values]).T
                df1.columns = ["trade_date", stock_code]
                df2.columns = ["trade_date", stock_code]
                df1 = df1.drop_duplicates(['trade_date'], keep='last')
                df2 = df2.drop_duplicates(['trade_date'], keep='last')
                df1 = df1.dropna(axis=0)
                df2 = df2.dropna(axis=0)
                df1 = pd.merge(df1, tradingday, on="trade_date",
                               how="outer", sort=True)
                df2 = pd.merge(df2, tradingday, on="trade_date",
                               how="outer", sort=True)
                df1 = df1.fillna(method="ffill").fillna(0)
                df2 = df2.fillna(method="ffill").fillna(0)
                df1 = pd.merge(df1, tradingday, on="trade_date",
                               how="inner", sort=True)
                df2 = pd.merge(df2, tradingday, on="trade_date",
                               how="inner", sort=True)
                cagrpni_PY5[stock_code] = df1[stock_code]
                cagrgr_PY5[stock_code] = df2[stock_code]
            except Exception:
                print(stock_code)
        cagrpni_PY5.index = tradingday['trade_date']
        cagrpni_PY5.columns = [x[0:6] for x in stock_list]
        cagrpni_PY5 = cagrpni_PY5 / 100
        self.save_h5(cagrpni_PY5, "cagrpni_PY5")
        cagrgr_PY5.index = tradingday['trade_date']  # 增长率采用小数形式表示
        cagrgr_PY5.columns = [x[0:6] for x in stock_list]
        cagrgr_PY5 = cagrgr_PY5 / 100
        self.save_h5(cagrgr_PY5, "cagrgr_PY5")
        # ======================================================================
        # ======================================================================
        data = c.tradedates(Start_Date, End_Date,
                            "period=1,order=1,market=CNSESH")
        tradingday_o = pd.DataFrame([str(parse(x))[0:4] + str(parse(x))[5:7] + str(
            parse(x))[8:10] for x in data.Data], columns=["trade_date"])
        """一致预测每股收益(FY1)、一致预测归属母公司净利润(FY1)、一致预测归属母公司净利润(FY3)"""
        eps_FY1 = pd.DataFrame(
            np.ones((len(tradingday_o), len(stock_list))) * np.nan)
        sestni_FY1 = pd.DataFrame(
            np.ones((len(tradingday_o), len(stock_list))) * np.nan)
        sestni_FY3 = pd.DataFrame(
            np.ones((len(tradingday_o), len(stock_list))) * np.nan)
        for i, trade_date in enumerate(tradingday_o["trade_date"].tolist()):
            for _ in range(retry_count):
                try:
                    data_eps = c.css(stock_list, "SESTEPSFY1",
                                     "EndDate=" + trade_date + ",Ispandas=1")
                    eps_FY1.iloc[i, :] = data_eps["SESTEPSFY1"].values
                    data_sestni = c.css(
                        stock_list, "SESTNIFY1,SESTNIFY3", "EndDate=" + trade_date + ",Ispandas=1")
                    sestni_FY1.iloc[i, :] = data_sestni["SESTNIFY1"].values
                    sestni_FY3.iloc[i, :] = data_sestni["SESTNIFY3"].values
                    break
                except Exception:
                    time.sleep(pause)
        # ======================================
        eps_FY1.index = tradingday_o['trade_date']
        eps_FY1.columns = [x[0:6] for x in stock_list]
        self.save_h5(eps_FY1, "eps_FY1")
        # ======================================
        sestni_FY1.index = tradingday_o['trade_date']
        sestni_FY1.columns = [x[0:6] for x in stock_list]
        self.save_h5(sestni_FY1, "sestni_FY1")
        # ======================================
        sestni_FY3.index = tradingday_o['trade_date']
        sestni_FY3.columns = [x[0:6] for x in stock_list]
        self.save_h5(sestni_FY3, "sestni_FY3")
        return True

    # 根据下载的财报数据生成日频因子
    def factor_generator(self, indicator):
        """
        :return:
        """
        stock_list = os.listdir("E:\\AlphaPlatform\\FinanceReport\\")
        trading_day = pd.DataFrame(pro.trade_cal(exchange='SSE',
                                                 start_date="19960101",
                                                 end_date="20200108",
                                                 is_open=1)["cal_date"].tolist(),
                                                 columns=["TradeDate"])
        fina_data = []
        for code in stock_list:
            print(code)
            report_list = os.listdir("E:\\AlphaPlatform\\FinanceReport\\" + code)
            stock_data = []
            for report in report_list:
                stock_data.append(pd.read_hdf(
                    "E:\\AlphaPlatform\\FinanceReport\\" + code + "\\" + report, report[:-3]).loc[indicator])
            stock_data = pd.DataFrame(stock_data).iloc[:, 0:2].rename(columns={
                "STMTACTDATE": "TradeDate"}).fillna(method="ffill").drop_duplicates(subset="TradeDate", keep="last")
            stock_data.TradeDate = stock_data.TradeDate.apply(
                lambda x: datetime.strftime(parse(x), "%Y%m%d"))
            stock_data = pd.merge(stock_data, trading_day, how="outer", on="TradeDate").sort_values(
                by="TradeDate").fillna(method="ffill")
            fina_data.append(pd.merge(stock_data, trading_day, how="inner",
                                on="TradeDate").iloc[:, 0])
        df = pd.DataFrame(fina_data).T
        df.columns = [x[0:6] for x in df.columns.tolist()]
        df.index = trading_day
        self.save_h5(df, indicator)
    print()

    # 根据已有日频因子生成衍生因子
    def factor_calculate(self):
        Start_Date = self.Start_Date
        End_Data = self.End_Date
        log = self.log
        sestni_FY0 = pd.read_hdf(
            FactorPath + "FinaFactor\\" + "sestni_FY0.h5", "sestni_FY0")
        col = sestni_FY0.columns.tolist()
        sestni_FY1 = pd.read_hdf(
            FactorPath + "FinaFactor\\" + "sestni_FY1.h5", "sestni_FY1").reindex(columns=col)
        sestni_FY3 = pd.read_hdf(
            FactorPath + "FinaFactor\\" + "sestni_FY3.h5", "sestni_FY3").reindex(columns=col)
        # 未来一年一致预测净利润增长率
        sestni_YOY1 = (np.divide(sestni_FY1, sestni_FY0) - 1)
        sestni_YOY1 = sestni_YOY1.fillna(
            method="ffill").fillna(0).replace([-1], [0])
        self.save_h5(sestni_YOY1, "sestni_YOY1")
        # 未来三年一致预测净利润增长率
        sestni_YOY3 = pow(np.divide(sestni_FY3, sestni_FY0), 1 / 3) - 1
        sestni_YOY3 = sestni_YOY3.fillna(
            method="ffill").fillna(0).replace([-1], [0])
        self.save_h5(sestni_YOY3, "sestni_YOY3")
        """衍生财务因子就算和保存"""

        return


if __name__ == "__main__":
    """建立数据连接"""
    import tushare as ts

    pro = ts.pro_api()
    """设置下载日期"""
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    trading_day = [str(x) for x in TradeDate['trade_date']]
    Start_Date = str(trading_day[0])
    End_Date = str(trading_day[-1])
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 设置日志文件
    log = Logger(Pre_path + "\\Logging\\C_DataUpdate_Fina\\" +
                 'Finance_Error.log',
                 level='info')
    print("Data of today begin to update !")
    data_update = Data_Update(Start_Date, End_Date, log)
    # data_update.finance_data()
    data_update.factor_generator("ROA")
    data_update.factor_calculate()
    data_update.factor_update()
    print("The update of finance_data is finished~")
