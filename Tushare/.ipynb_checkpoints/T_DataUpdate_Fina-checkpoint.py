#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/19 20:27
# @Author: Tu Xinglong
# @File  : T_DataUpdate_Fina.py

import os
import sys

import numpy as np
import pandas as pd
import tushare as ts
import warnings
import time
# import datetime as dt
import multiprocessing

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.abspath(os.path.dirname(CurrentPath))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
DataPath = Pre_path + '\\MarketData\\'
InfoPath = Pre_path + '\\MarketInfo\\'
FactorPath = Pre_path + '\\FactorData\\'

from LoggingPlus import Logger  # noqa


class Data_Update(object):
    '''
    财务数据：三大报表、业绩预报、业绩快报
    市场参考数据：港股通、深股通、融资融券
    其他数据：大宗交易、股票质押、限售股解禁
    '''

    def __init__(self, startdate, enddate, pro, log):
        '''
        定义起始日期、数据库接口
        tushare接口初始化.
        '''
        self.Start_Date = startdate
        self.End_Date = enddate
        self.pro = pro
        self.log = log
        # 每周五或周末做一次全市场历史数据更新
        if (time.strftime("%A", time.localtime(time.time()))
                in ["Friday","Saturday","Sunday"]):
            self.FULL_UPDATE = True
        else:
            self.FULL_UPDATE = False

    def basic_data(self, retry_count=50, pause=5):
        '''
        股票交易日
        股票列表
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log

        # 获取股票交易日
        S_tradingDay = pro.trade_cal(exchange='SSE',
                                     start_date=Start_Date,
                                     end_date=End_Date,
                                     is_open=1)
        stock_day = S_tradingDay['cal_date'].tolist()

        # 当前上市交易股票列表
        Stock_Ref = pro.stock_basic(exchange='',
                                    list_status='L',
                                    fields='ts_code,symbol,name,area,industry,\
            list_date,market,exchange,is_hs')
        stock_list = Stock_Ref['ts_code'].tolist()

        return (stock_day, stock_list)

    def finance_data(self, trading_day, stock_list, retry_count=100, pause=5):
        '''
        利润表
        资产负债表
        现金流量表
        财务数据指标
        '''
        StartDate = self.Start_Date
        EndDate = self.End_Date
        pro = self.pro
        log = self.log
        trading_day = pd.DataFrame(trading_day)
        trading_day.columns = ['trade_date']
        if (True):  # 每到财报日重新运行一次
            try:
                for i, stock_code in enumerate(stock_list):
                    """利润表"""
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            income_data = pro.income(ts_code=stock_code,start_date=StartDate,end_date=EndDate,fields=
                                                     'ts_code,ann_date,end_date,report_type,comp_type,total_profit')
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if income_data is None:
                                continue
                            else:
                                break
                    if income_data.empty:
                        pass
                    else:
                        income_data = income_data.sort_values(by='end_date')
                        income_data = income_data.drop_duplicates(['ann_date'],keep='last')
                        income_data = pd.merge(income_data,trading_day,left_on='ann_date',right_on='trade_date',how='outer',sort=True)
                        income_data['ts_code'] = income_data['ts_code'].fillna(method='ffill').fillna(method='bfill')
                        income_data = income_data.fillna(method='ffill').fillna(0)
                        income_data = pd.merge(income_data,trading_day,on='trade_date',how='inner',sort=True)
                        income_data = income_data.drop_duplicates(['trade_date'])
                        income_data.index = range(income_data.shape[0])
                        trade_date = income_data.pop('trade_date')
                        income_data.insert(1, 'trade_date', trade_date)
                        income_data.to_csv(InfoPath + "FinanceInfo\\InCome\\" + stock_code[0:6] + ".csv", index=False, mode='w')
                    # ==============================================================
                    # ==============================================================
                    """资产负债表"""
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            balancesheet_data = pro.balancesheet(ts_code=stock_code,start_date=StartDate,end_date=EndDate,
                                                                 fields='ts_code,ann_date,end_date,total_ncl,total_assets,total_liab')
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if balancesheet_data is None:
                                continue
                            else:
                                break
                    if balancesheet_data.empty:
                        pass
                    else:
                        balancesheet_data = balancesheet_data.sort_values(by='end_date')
                        balancesheet_data = balancesheet_data.drop_duplicates(['ann_date'],keep='last')
                        balancesheet_data = pd.merge(balancesheet_data,trading_day,left_on='ann_date',right_on='trade_date',how='outer',sort=True)
                        balancesheet_data['ts_code'] = balancesheet_data['ts_code'].fillna(method='ffill').fillna(method='bfill')
                        balancesheet_data = balancesheet_data.fillna(method='ffill').fillna(0)
                        balancesheet_data = pd.merge(balancesheet_data,trading_day,on='trade_date',how='inner',sort=True)
                        balancesheet_data = balancesheet_data.drop_duplicates(['trade_date'])
                        balancesheet_data.index = range(balancesheet_data.shape[0])
                        trade_date = balancesheet_data.pop('trade_date')
                        balancesheet_data.insert(1, 'trade_date', trade_date)
                        balancesheet_data['total_equity'] = balancesheet_data['total_assets']-balancesheet_data['total_liab']
                        balancesheet_data.to_csv(InfoPath + "FinanceInfo\\BalanceSheet\\" + stock_code[0:6] + ".csv", index=False, mode='w')
                    # ==============================================================
                    # ==============================================================
                    """现金流量表"""
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            cashflow_data = pro.cashflow(ts_code=stock_code,start_date=StartDate,end_date=EndDate,
                                                         fields='ts_code,ann_date,end_date,net_profit')
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if cashflow_data is None:
                                continue
                            else:
                                break
                    if cashflow_data.empty:
                        pass
                    else:
                        cashflow_data = cashflow_data.sort_values(by='end_date')
                        cashflow_data = cashflow_data.drop_duplicates(['ann_date'],keep='last')
                        cashflow_data = pd.merge(cashflow_data,trading_day,left_on='ann_date',right_on='trade_date',how='outer',sort=True)
                        cashflow_data['ts_code'] = cashflow_data['ts_code'].fillna(method='ffill').fillna(method='bfill')
                        cashflow_data = cashflow_data.fillna(method='ffill').fillna(0)
                        cashflow_data = pd.merge(cashflow_data,trading_day,on='trade_date',how='inner',sort=True)
                        cashflow_data = cashflow_data.drop_duplicates(['trade_date'])
                        cashflow_data.index = range(cashflow_data.shape[0])
                        trade_date = cashflow_data.pop('trade_date')
                        cashflow_data.insert(1, 'trade_date', trade_date)
                        cashflow_data.to_csv(InfoPath + "FinanceInfo\\CashFlow\\" + stock_code[0:6] + ".csv",index=False, mode='w')
                    # ===========================================================
                    # ===========================================================
                    """财务指标数据"""
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            finaindicator_data = pro.fina_indicator(ts_code=stock_code,start_date=StartDate,end_date=EndDate,
                                                                    fields='ts_code,ann_date,end_date,eps,bps,cfps,roe,roa,debt_to_assets')
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if finaindicator_data is None:
                                continue
                            else:
                                break
                    if finaindicator_data.empty:
                        pass
                    else:
                        finaindicator_data = finaindicator_data.sort_values(by='end_date')
                        finaindicator_data = finaindicator_data.drop_duplicates(['ann_date'],keep='last')
                        finaindicator_data = pd.merge(finaindicator_data,trading_day,left_on='ann_date',right_on='trade_date',how='outer',sort=True)
                        finaindicator_data['ts_code'] = finaindicator_data['ts_code'].fillna(method='ffill').fillna(method='bfill')
                        finaindicator_data = finaindicator_data.fillna(method='ffill').fillna(0)
                        finaindicator_data = pd.merge(finaindicator_data,trading_day,on='trade_date',how='inner',sort=True)
                        finaindicator_data = finaindicator_data.drop_duplicates(['trade_date'])
                        finaindicator_data.index = range(finaindicator_data.shape[0])
                        trade_date = finaindicator_data.pop('trade_date')
                        finaindicator_data.insert(1, 'trade_date', trade_date)
                        finaindicator_data.to_csv(InfoPath + "FinanceInfo\\FinaIndicator\\" +stock_code[0:6] + ".csv",index=False,mode='w')
                    # ===========================================================
                    # ===========================================================
            except BaseException:
                log.logger.error(stock_code + " meets a serious error and needs to fill")
        else: #增量更新目前用不到
            pass
        return

    def generate_factor(self):
        '''
        合成因子数据：价量因子，财务因子等
        pricefactor: open、close、high、low、open_qfq、close_qfq、high_qfq、low_qfq
                     adjfactor、amount、volume、return、turnover_rate、volume_ratio
        finfactor  : pe_ttm、ps_ttm、total_mv、circ_mv、、、
        '''
        # =========================================================
        # =========================================================
        """价量因子数据"""
        open            = []  # 开盘价
        close           = []  # 收盘价
        high            = []  # 最高价
        low             = []  # 最低价
        open_qfq        = []  # 前复权开盘价
        close_qfq       = []  # 前复权收盘价
        high_qfq        = []  # 前复权最高价
        low_qfq         = []  # 前复权最低价
        open_hfq        = []  # 后复权开盘价
        close_hfq       = []  # 后复权收盘价
        amount          = []  # 成交量
        volume          = []  # 成交额
        ret             = []  # 个股收益率
        turnover_rate   = []  # 换手率
        pe_ttm          = []  # 市盈率TTM
        total_mv        = []  # 总市值
        """财务因子数据"""
        total_ncl       = []  # 非流动负债合计
        total_assets    = []  # 总资产
        total_liab      = []  # 总负债
        total_equity    = []  # 总权益
        cfps            = []  # 每股自由现金流净额
        # 读取文件列表--股票代码
        files_list = os.listdir(InfoPath + "StockInfo\\DayQuota\\")
        stock_code = [x[0:6] for x in files_list]
        for stock_name in files_list:
            if stock_name[0] != str(6):
                stock_day = pd.read_csv(DataPath + "Day\\SZE\\" + stock_name)
            else:
                stock_day = pd.read_csv(DataPath + "Day\\SSE\\" + stock_name)
            stock_quota = pd.read_csv(InfoPath + "StockInfo\\DayQuota\\" + stock_name)
            open.append(stock_day['open'])
            close.append(stock_day['close'])
            high.append(stock_day['high'])
            low.append(stock_day['low'])
            open_qfq.append(stock_day['open_qfq'])
            close_qfq.append(stock_day['close_qfq'])
            high_qfq.append(stock_day['high_qfq'])
            low_qfq.append(stock_day['low_qfq'])
            open_hfq.append(stock_day['open_hfq'])
            close_hfq.append(stock_day['close_hfq'])
            amount.append(stock_day['amount'])
            volume.append(stock_day['vol'])
            ret.append(stock_day['ret'])
            turnover_rate.append(stock_quota['turnover_rate'])
            pe_ttm.append(stock_quota['pe_ttm'])
            total_mv.append(stock_quota['total_mv'])
            del stock_day,stock_quota
            #=================================================
            #=================================================
            balancesheet_data = pd.read_csv(InfoPath + "FinanceInfo\\BalanceSheet\\" + stock_name)
            finaindicator_data = pd.read_csv(InfoPath + "FinanceInfo\\FinaIndicator\\" + stock_name)
            total_ncl.append(balancesheet_data['total_ncl'])
            total_assets.append(balancesheet_data['total_assets'])
            total_liab.append(balancesheet_data['total_liab'])
            total_equity.append(balancesheet_data['total_equity'])
            cfps.append(finaindicator_data['cfps'])
            del balancesheet_data,finaindicator_data
        stock_day = pd.read_csv(DataPath + "Day\\SZE\\" + "000001.csv")
        trade_date = stock_day['trade_date']
        factor_list = [open, close, high, low, open_qfq, close_qfq, high_qfq,
            low_qfq, open_hfq, close_hfq, amount, volume, ret, turnover_rate,pe_ttm, total_mv,total_ncl,
            total_assets,total_liab,total_equity,cfps]
        factor_name = [
            'open', 'close', 'high', 'low', 'open_qfq', 'close_qfq','high_qfq',
            'low_qfq', 'open_hfq','close_hfq','amount', 'volume', 'ret','turnover_rate', 'pe_ttm', 'total_mv',
            'total_ncl','total_assets','total_liab','total_equity','cfps']
        try:
            for i, factor in enumerate(factor_list):
                factor = pd.DataFrame(factor).T
                factor.index = trade_date
                factor.columns = stock_code
                factor.to_hdf(FactorPath + "FinaFactor\\" + factor_name[i] + ".h5",factor_name[i], mode='w')
                del factor
        except BaseException:
            log.logger.error(("因子" + factor_name[i] + "合成错误,请检查!"))
        # ==================================================================
        # ==================================================================
        """指数因子数据"""
        index = []
        index_file = [x for x in os.listdir(DataPath + "Day\\INDEX\\")]
        index_code = [i[0:6] for i in index_file]
        for m, filename in enumerate(index_file):
            index_day = pd.read_csv(DataPath + "Day\\INDEX\\" + filename)
            index.append(index_day['open'])
            index.append(index_day['close'])
        index = pd.DataFrame(index).T
        index.index = index_day['trade_date']
        index.columns = ['上证综指open', '上证综指close', '上证50open', '上证50close','中证1000open',
                         '中证1000close', '中证100open', '中证100close','中证200open', '中证200close',
                         '中证500open', '中证500close','中证800open', '中证800close', '深证成指open',
                         '深证成指close','中小板open', '中小板close', '创业板open', '创业板close',
                         '沪深300open', '沪深300close']
        index.to_hdf(FactorPath + "FinaFactor\\" + "index" + ".h5", "index",mode='w')

        return

def fina_func(params):
    pro = ts.pro_api()
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 建立日志对象
    log = Logger(Pre_path + "\\Logging\\T_DataUpdate_Fina\\" +
                 'Finance_Error.log',
                 level='info')
    start_date = params['start_date']
    end_date = params['end_date']
    trading_day = params['trading_day']
    data_update = Data_Update(start_date, end_date, pro, log)
    stock_list = params['stock_list']
    data_update.finance_data(trading_day, stock_list)

    return


def set_params(code_list, trading_day, start_date, end_date):
    td = {
        'trading_day': trading_day,
        'start_date': start_date,
        'end_date': end_date,
    }
    params = []
    for i, sec_code in enumerate(code_list):
        td['stock_list'] = sec_code
        params.append(td.copy())
    return params


if __name__ == "__main__":
    """初始化tushare接口"""
    pro = ts.pro_api()
    """建立数据连接"""
    # db = MySQLConn.DBCONN()
    """设置下载日期"""
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    Start_Date = "19951231" #str(TradeDate.iloc[0, 0])
    End_Date = str(TradeDate.iloc[-1,0])
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 建立日志对象
    log = Logger(Pre_path + "\\Logging\\T_DataUpdate_Fina\\" +'Finance_Error.log',level='info')
    # 开始更新数据
    data_update = Data_Update(Start_Date, End_Date, pro, log)
    trading_day, stock_list = data_update.basic_data()
    print("Data of finance_data begin to update !")
    # =======================================================
    # =======================================================
    """更新财务数据"""
    print("The update of basic_data is finished~")
    stock_list_o = [x[0:6] + ".SZ" for x in os.listdir(DataPath + "\\Day\\SZE\\")] \
                + [i[0:6] + ".SH" for i in os.listdir(DataPath + "\\Day\\SSE\\")]
    stock_list = [stock_list_o[i:i + 50] for i in range(0, len(stock_list_o), 50)]
    para_x = set_params(stock_list, trading_day, Start_Date, End_Date)
    pool = multiprocessing.Pool(8)
    # 多进程map中不能有传入地址参数pro、log
    pool.map(fina_func, para_x)
    pool.close()
    pool.join()
    print("The update of finance_data is finished~")
    # =========================================================
    # =========================================================
    """更新因子数据"""
    data_update.generate_factor()
    print("The update of factor_data is finished~")
