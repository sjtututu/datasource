#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/8 16:17
# @Author: Tu Xinglong
# @File  : T_DataUpdate_Day.py

import os
import sys
import re
import shutil

import numpy as np
import pandas as pd
import tushare as ts
import warnings
import time
from dateutil.parser import parse
import datetime as dt
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
    股票行情信息数据获取类
    tushare接口调取数据需要注意的问题：
    1.交易日里与行情数据的日期顺序相反
    2.一次性调取数据的量不能超过4000条
    3.需要高级权限才能调取分钟K线及指数权重
    4.返回的数据剔除了停牌日，需自行补上填充
    5.需要付费才能进入tushare高级用户群
    6.每天更新数据之前判断是否有新股
    7.更新速度与网速、磁盘读取速度有关
    8.因ts限制，有时部分数据没跟更新需补上
    '''

    def __init__(self, startdate, enddate, pro, log):
        '''
        定义起始日期、数据库接口
        tushare接口初始化
        '''
        self.Start_Date = startdate
        self.End_Date = enddate
        self.pro = pro
        self.log = log
        # self.conn = db

    def stock_basic(self):
        '''
        股票基础数据下载部分
        包括交易日、日历日、股票列表等
        w:文件覆盖模式
        a:文件追加模式
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log

        # 获取交易日
        Trading_Day = pro.trade_cal(exchange='SSE',
                                    start_date=Start_Date,
                                    end_date=End_Date,
                                    is_open=1)
        Trading_Day.to_csv(InfoPath + "StockInfo\\BasicData\\Trading_Day.csv",
                           index=False,
                           header=False,
                           mode='a')

        # 当前上市交易股票列表
        Stock_Ref = pro.stock_basic(exchange='',
                                    list_status='L',
                                    fields='ts_code,symbol,\
            name,area,industry,list_date,market,exchange,is_hs')
        Stock_Ref.to_csv(InfoPath + "StockInfo\\BasicData\\Stock_Ref.csv",
                         index=False,
                         mode='w',
                         encoding='GBK')

        # 沪股通成分股
        SSE_Tong = pro.hs_const(hs_type='SH')
        SSE_Tong.to_csv(InfoPath + "StockInfo\\BasicData\\SSE_Tong.csv",
                        index=False,
                        mode='w')

        # 深股通成分股
        SZE_Tong = pro.hs_const(hs_type='SZ')
        SZE_Tong.to_csv(InfoPath + "StockInfo\\BasicData\\SZE_Tong.csv",
                        index=False,
                        mode='w')

        # 股票改名信息
        Stockname_Change = pro.namechange(
            ts_code='',
            fields='ts_code,name,start_date,end_date,change_reason')
        Stockname_Change.to_csv(InfoPath +
                                "StockInfo\\BasicData\\Stockname_Change.csv",
                                index=False,
                                mode='w',
                                encoding='GBK')

        # 上市公司基本信息
        Stockcompany_Info = pro.stock_company(exchange='',
                                              fields='ts_code,\
            chairman,manager,secretary,reg_capital,setup_date,province')
        Stockcompany_Info.to_csv(InfoPath +
                                 "StockInfo\\BasicData\\Stockcompany_Info.csv",
                                 index=False,
                                 mode='w',
                                 encoding='GBK')

        # 新股上市信息
        Stock_New = pro.new_share(start_date=Start_Date, end_date=End_Date)
        Stock_New = Stock_New.sort_values(by='ipo_date')
        Stock_New.to_csv(InfoPath + "StockInfo\\BasicData\\Stock_New.csv",
                         index=False,
                         header=False,
                         mode='a',encoding='GBK')
        return Stock_Ref['ts_code']

    def stock_day(self, trading_day, stock_list, retry_count=100, pause=5):
        '''
        股票日线行情数据下载部分
        高开低收,成交量,成交额,复权数据
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log
        trading_day = pd.DataFrame(trading_day)
        trading_day.columns = ['trade_date']
        if (False):  # 首次更新
            for i, Stock_code in enumerate(stock_list):
                """获取复权因子和股票日线行情"""
                try:
                    # 获取复权因子
                    adj_factor = pro.adj_factor(ts_code=Stock_code,
                                                trade_date='',
                                                start_date=Start_Date,
                                                end_date=End_Date)
                    # 对停牌日进行合并
                    adj_factor = pd.merge(adj_factor,
                                          trading_day,
                                          on='trade_date',
                                          how='outer',
                                          sort=True)
                    # 股票行情数据
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            Day_data = pro.daily(ts_code=Stock_code,
                                                 start_date=Start_Date,
                                                 end_date=End_Date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if Day_data is None:
                                continue
                            else:
                                break
                    # 对停牌日进行合并
                    Day_data = pd.merge(Day_data,
                                        trading_day,
                                        on='trade_date',
                                        how='outer',
                                        sort=True)
                    # 后复权
                    df_hfq = Day_data[['open', 'high', 'low',
                                       'close']].mul(adj_factor['adj_factor'],
                                                     axis=0)
                    df_hfq.columns = [
                        'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'
                    ]
                    # 前复权
                    df_qfq = Day_data[['open', 'high', 'low', 'close']].mul(
                        adj_factor['adj_factor'],
                        axis=0) / adj_factor['adj_factor'].iloc[-1]
                    df_qfq.columns = [
                        'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq'
                    ]
                    # 计算收益率
                    df_ret = pd.DataFrame(df_qfq['close_qfq'].pct_change())
                    df_ret.columns = ['ret']
                    # 填充计算收益率产生的inf值
                    df_ret = df_ret.replace([np.inf, -np.inf], [0, 0])
                    # 合并数据到一个表中
                    Day_data = pd.concat([
                        Day_data, df_qfq, df_hfq, df_ret,
                        adj_factor['adj_factor']
                    ],
                                         axis=1)
                    # 停牌股票成交量、成交额为0
                    Day_data[['change', 'pct_chg', 'vol', 'amount',
                              'ret']] = Day_data[[
                                  'change', 'pct_chg', 'vol', 'amount', 'ret'
                              ]].fillna(0)
                    # 其他行情信息向前填充
                    Day_data['ts_code'] = Day_data['ts_code'].fillna(
                        method='ffill').fillna(method='bfill')
                    Day_data = Day_data.fillna(method='ffill').fillna(0)
                    # 保存数据
                    if Stock_code[7:9] == "SZ":
                        Day_data.to_csv(
                            DataPath + "Day\\SZE\\" + Stock_code[0:6] + ".csv",
                            index=False,
                            # header=False,
                            mode='w')
                    else:
                        Day_data.to_csv(
                            DataPath + "Day\\SSE\\" + Stock_code[0:6] + ".csv",
                            index=False,
                            # header=False,
                            mode='w')
                    """每日指标更新"""
                    ######################################
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            Day_quato = pro.daily_basic(
                                ts_code=Stock_code,
                                start_date=Start_Date,
                                end_date=End_Date,
                                fields=
                                'ts_code,trade_date,turnover_rate,turnover_rate_f,\
                                volume_ratio, pe,pe_ttm,pb,ps,ps_ttm,total_share,\
                                float_share,free_share,total_mv,circ_mv')
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if Day_quato is None:
                                continue
                            else:
                                break
                        # 每日指标包含非交易日，但不包含停牌日，
                    # 所以需先取交集，再取并集
                    Day_quato = pd.merge(Day_quato,
                                         trading_day,
                                         on='trade_date',
                                         how='outer',
                                         sort=True)
                    Day_quato = pd.merge(Day_quato,
                                         trading_day,
                                         on='trade_date',
                                         how='inner',
                                         sort=True)
                    # 其他行情信息向前填充
                    Day_quato[[
                        'turnover_rate', 'turnover_rate_f', 'volume_ratio'
                    ]] = Day_quato[[
                        'turnover_rate', 'turnover_rate_f', 'volume_ratio'
                    ]].fillna(0)
                    Day_quato['ts_code'] = Day_quato['ts_code'].fillna(
                        method='ffill').fillna(method='bfill')
                    Day_quato = Day_quato.fillna(method='ffill').fillna(0)
                    Day_quato = Day_quato.reindex(columns=[
                        'ts_code', 'trade_date', 'turnover_rate',
                        'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm',
                        'pb', 'ps', 'ps_ttm', 'total_share', 'float_share',
                        'free_share', 'total_mv', 'circ_mv'
                    ])
                    if Day_quato.empty:
                        pass
                    else:
                        # 保存数据
                        Day_quato.to_csv(
                            InfoPath + "StockInfo\\DayQuota\\" +
                            Stock_code[0:6] + ".csv",
                            index=False,
                            # header=False,
                            mode='w')
                    """个股资金流向"""
                    #############################################
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            Money_flow = pro.moneyflow(ts_code=Stock_code,
                                                       start_date=Start_Date,
                                                       end_date=End_Date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            if Money_flow is None:
                                continue
                            else:
                                break
                    Money_flow = pd.merge(Money_flow,
                                          trading_day,
                                          on='trade_date',
                                          how='outer',
                                          sort=True)
                    # 其他信息向前填充
                    Money_flow['ts_code'] = Money_flow['ts_code'].fillna(
                        method='ffill').fillna(method='bfill')
                    Money_flow = Money_flow.fillna(0)
                    # 个股资金流量直接追加模式
                    Money_flow.to_csv(
                        InfoPath + "StockInfo\\MoneyFlow\\" + Stock_code[0:6] +
                        ".csv",
                        index=False,
                        # header=False,
                        mode='a')
                except BaseException:
                    log.logger.error(" Logged " + Stock_code)

        # ==================================================================
        # ==================================================================

        else:  # 增量更新
            for i, Stock_code in enumerate(stock_list):
                try:
                    # =======================================================
                    # =======================================================
                    """获取复权因子和股票日线行情增量跟新"""
                    if (not os.path.exists(  # 判断是否有新股
                        (DataPath + "Day\\SZE\\" + Stock_code[0:6] +
                         ".csv"))) and (not os.path.exists(
                             (DataPath + "Day\\SSE\\" + Stock_code[0:6] +
                              ".csv"))):
                        # 用来判断当前数据长度
                        shape_data = pd.read_csv(DataPath + "Day\\SZE\\" +
                                                 "000001" + ".csv")
                        Data_zeros = pd.DataFrame(
                            np.zeros((shape_data.shape[0] - 1,
                                      shape_data.shape[1])),
                            columns=[
                                'ts_code', 'trade_date', 'open', 'high', 'low',
                                'close', 'pre_close', 'change', 'pct_chg',
                                'vol', 'amount', 'adj_factor', 'open_qfq',
                                'high_qfq', 'low_qfq', 'close_qfq', 'open_hfq',
                                'high_hfq', 'low_hfq', 'close_hfq', 'ret'
                            ])
                        # 获取复权因子
                        adj_factor = pro.adj_factor(ts_code=Stock_code,
                                                    trade_date='',
                                                    start_date=Start_Date,
                                                    end_date=End_Date)
                        # 股票行情数据
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                Day_data = pro.daily(ts_code=Stock_code,
                                                     start_date=Start_Date,
                                                     end_date=End_Date)
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if Day_data is None:
                                    continue
                                else:
                                    break
                        data_temp = pd.DataFrame([0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 index=[
                                                     'open_qfq', 'high_qfq',
                                                     'low_qfq', 'close_qfq',
                                                     'open_hfq', 'high_hfq',
                                                     'low_hfq', 'close_hfq',
                                                     'ret'
                                                 ]).T
                        if Day_data.empty:
                            pass
                        else:
                            Day_data['adj_factor'] = adj_factor['adj_factor']
                            Day_data = pd.concat([Day_data, data_temp], axis=1)
                            Day_data = pd.concat([Data_zeros, Day_data],
                                                 axis=0)
                            # 填充股票代码
                            Day_data['ts_code'] = Stock_code
                            Day_data.index = range(Day_data.shape[0])
                            # 填充交易日期
                            Day_data['trade_date'] = shape_data['trade_date']
                    else:
                        # 停牌日复权因子依然有数据
                        # 股票日线行情则返回空值
                        # 获取复权因子
                        adj_factor = pro.adj_factor(ts_code=Stock_code,
                                                    trade_date='',
                                                    start_date=Start_Date,
                                                    end_date=End_Date)
                        # 对停牌日进行合并
                        adj_factor = pd.merge(adj_factor,
                                              trading_day,
                                              on='trade_date',
                                              how='outer',
                                              sort=True)
                        # 股票行情数据
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                Day_data = pro.daily(ts_code=Stock_code,
                                                     start_date=Start_Date,
                                                     end_date=End_Date)
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if Day_data is None:
                                    continue
                                else:
                                    break
                        # 截止昨日股票数据
                        if Stock_code[7:9] == "SZ":
                            Data_yesterday = pd.read_csv(DataPath +
                                                         "Day\\SZE\\" +
                                                         Stock_code[0:6] +
                                                         ".csv")
                        else:
                            Data_yesterday = pd.read_csv(DataPath +
                                                         "Day\\SSE\\" +
                                                         Stock_code[0:6] +
                                                         ".csv")
                        # 纠正表格列名顺序
                        Data_yesterday = Data_yesterday.reindex(columns=[
                            'ts_code', 'trade_date', 'open', 'high', 'low',
                            'close', 'pre_close', 'change', 'pct_chg', 'vol',
                            'amount', 'adj_factor'
                        ])
                        # 对股票代码进行填充
                        # Data_yesterday['ts_code'] = Data_yesterday[
                        #     'ts_code'].iloc[-1]
                        # 增量更新天数间隔判断
                        if (int(str(
                            (parse(End_Date) - parse(Start_Date)).days))) == 0:
                            # 判断该股票当日是否停牌或退市
                            if Day_data.empty:
                                Day_data = pd.DataFrame(
                                    [
                                        Stock_code, End_Date, np.nan, np.nan,
                                        np.nan, np.nan, np.nan, 0, 0, 0, 0,
                                        np.nan
                                    ],
                                    index=[
                                        'ts_code', 'trade_date', 'open',
                                        'high', 'low', 'close', 'pre_close',
                                        'change', 'pct_chg', 'vol', 'amount',
                                        'adj_factor'
                                    ]).T
                                Day_data = pd.concat(
                                    [Data_yesterday, Day_data], axis=0)
                                Day_data = Day_data.fillna(method="ffill")
                            else:
                                Day_data['adj_factor'] = adj_factor[
                                    'adj_factor']
                                Day_data = pd.concat(
                                    [Data_yesterday, Day_data], axis=0)
                        else:
                            # 对停牌日进行合并
                            Day_data = pd.merge(Day_data,
                                                trading_day,
                                                on='trade_date',
                                                how='outer',
                                                sort=True)
                            Day_data['adj_factor'] = adj_factor['adj_factor']
                            Day_data = pd.concat([Data_yesterday, Day_data],
                                                 axis=0)
                        Day_data.index = range(Day_data.shape[0])
                        # 后复权
                        df_hfq = Day_data[['open', 'high', 'low', 'close'
                                           ]].mul(Day_data['adj_factor'],
                                                  axis=0)
                        df_hfq.columns = [
                            'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'
                        ]
                        # 前复权
                        df_qfq = Day_data[[
                            'open', 'high', 'low', 'close'
                        ]].mul(Day_data['adj_factor'],
                               axis=0) / Day_data['adj_factor'].iloc[-1]
                        df_qfq.columns = [
                            'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq'
                        ]
                        # 计算收益率
                        df_ret = pd.DataFrame(df_qfq['close_qfq'].pct_change())
                        df_ret.columns = ['ret']
                        # 填充计算收益率产生的inf值
                        df_ret = df_ret.replace([np.inf, -np.inf], [0, 0])
                        # 合并数据到一个表中
                        Day_data = pd.concat(
                            [Day_data, df_qfq, df_hfq, df_ret], axis=1)
                        # 停牌股票成交量、成交额为0
                        Day_data[['change', 'pct_chg', 'vol', 'amount',
                                  'ret']] = Day_data[[
                                      'change', 'pct_chg', 'vol', 'amount',
                                      'ret'
                                  ]].fillna(0)
                        # 其他行情信息向前填充
                        Day_data['ts_code'] = Day_data['ts_code'].fillna(
                            method='ffill').fillna(method='bfill')
                        Day_data = Day_data.fillna(method='ffill').fillna(0)
                        # 去重和按照日期排序
                        Day_data = Day_data.drop_duplicates(subset=["trade_date"])
                        Day_data["trade_date"] = Day_data["trade_date"].astype('int64')
                        Day_data = Day_data.sort_values(by="trade_date")
                    if Day_data.empty:
                        pass
                    else:
                        # 保存数据
                        if Stock_code[7:9] == "SZ":
                            Day_data.to_csv(
                                DataPath + "Day\\SZE\\" + Stock_code[0:6] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='w')
                        else:
                            Day_data.to_csv(
                                DataPath + "Day\\SSE\\" + Stock_code[0:6] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='w')

                    # ==============================================================
                    # ==============================================================

                    """股票每日指标增量更新"""
                    # 以文件是否存在来判断是否为新股
                    if not os.path.exists(InfoPath + "StockInfo\\DayQuota\\" +
                                          Stock_code[0:6] +
                                          ".csv"):  # 文件不存在则为新股
                        # 用来判断当前数据长度
                        shape_data = pd.read_csv(InfoPath +
                                                 "StockInfo\\DayQuota\\" +
                                                 "000001" + ".csv")
                        Data_zeros = pd.DataFrame(
                            np.zeros((shape_data.shape[0] - 1,
                                      shape_data.shape[1])),
                            columns=[
                                'ts_code', 'trade_date', 'turnover_rate',
                                'turnover_rate_f', 'volume_ratio', 'pe',
                                'pe_ttm', 'pb', 'ps', 'ps_ttm', 'total_share',
                                'float_share', 'free_share', 'total_mv',
                                'circ_mv'
                            ])
                        # 每日指标
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                Day_quato = pro.daily_basic(
                                    ts_code=Stock_code,
                                    start_date=Start_Date,
                                    end_date=End_Date,
                                    fields=
                                    'ts_code,trade_date,turnover_rate,turnover_rate_f,\
                                    volume_ratio, pe,pe_ttm,pb,ps,ps_ttm,total_share,\
                                    float_share,free_share,total_mv,circ_mv')
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if Day_quato is None:
                                    continue
                                else:
                                    break
                        if Day_quato.empty:
                            pass
                        else:
                            Day_quato = pd.concat([Data_zeros, Day_quato],
                                                  axis=0)
                            Day_quato['ts_code'] = Stock_code
                            Day_quato.index = range(Day_quato.shape[0])
                            Day_quato['trade_date'] = shape_data['trade_date']
                            Day_quato = Day_quato.fillna(0)
                    else:  # 文件存在不为新股
                        Day_yesterday = pd.read_csv(InfoPath +
                                                    "StockInfo\\DayQuota\\" +
                                                    Stock_code[0:6] + ".csv")
                        # 对股票代码进行填充
                        Day_yesterday['ts_code'] = Day_yesterday[
                            'ts_code'].iloc[-1]
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                Day_quato = pro.daily_basic(
                                    ts_code=Stock_code,
                                    start_date=Start_Date,
                                    end_date=End_Date,
                                    fields=
                                    'ts_code,trade_date,turnover_rate,turnover_rate_f,\
                                    volume_ratio, pe,pe_ttm,pb,ps,ps_ttm,total_share,\
                                    float_share,free_share,total_mv,circ_mv')
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if Day_quato is None:
                                    continue
                                else:
                                    break
                        # 每日指标包含非交易日，但不包含停牌日，
                        # 所以需先取交集，再取并集
                        if Day_quato.empty:
                            Day_quato = pd.DataFrame(
                                [
                                    Stock_code, End_Date, 0, 0, 0, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan
                                ],
                                index=[
                                    'ts_code', 'trade_date', 'turnover_rate',
                                    'turnover_rate_f', 'volume_ratio', 'pe',
                                    'pe_ttm', 'pb', 'ps', 'ps_ttm',
                                    'total_share', 'float_share', 'free_share',
                                    'total_mv', 'circ_mv'
                                ]).T
                        else:
                            Day_quato = pd.merge(Day_quato,
                                                 trading_day,
                                                 on='trade_date',
                                                 how='outer',
                                                 sort=True)
                            Day_quato = pd.merge(Day_quato,
                                                 trading_day,
                                                 on='trade_date',
                                                 how='inner',
                                                 sort=True)
                        # 合并数据
                        Day_quato = pd.concat([Day_yesterday, Day_quato],
                                              axis=0)
                        Day_quato.index = range(Day_quato.shape[0])
                        # 其他行情信息向前填充
                        Day_quato[[
                            'turnover_rate', 'turnover_rate_f', 'volume_ratio'
                        ]] = Day_quato[[
                            'turnover_rate', 'turnover_rate_f', 'volume_ratio'
                        ]].fillna(0)
                        Day_quato['ts_code'] = Day_quato['ts_code'].fillna(
                            method='ffill').fillna(method='bfill')
                        Day_quato = Day_quato.fillna(method='ffill').fillna(0)
                        Day_quato = Day_quato.reindex(columns=[
                            'ts_code', 'trade_date', 'turnover_rate',
                            'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm',
                            'pb', 'ps', 'ps_ttm', 'total_share', 'float_share',
                            'free_share', 'total_mv', 'circ_mv'
                        ])
                        # 去重和重新排序
                        Day_quato["trade_date"] = Day_quato["trade_date"].astype('int64')
                        Day_quato = Day_quato.sort_values(by="trade_date")
                        Day_quato = Day_quato.drop_duplicates(subset=["trade_date"])
                    if Day_quato.empty:
                        pass
                    else:
                        # 保存数据
                        Day_quato.to_csv(
                            InfoPath + "StockInfo\\DayQuota\\" +
                            Stock_code[0:6] + ".csv",
                            index=False,
                            # header=False,
                            mode='w')

                    # ========================================================
                    # ========================================================
                    """个股资金流向"""
                    # 以文件是否存在来判断是否为新股
                    # 当日新股的资金流向可能为0,其新股文件
                    # 在后面的某一个交易日产生,不影响使用
                    if not os.path.exists(InfoPath + "StockInfo\\MoneyFlow\\" +
                                          Stock_code[0:6] +
                                          ".csv"):  # 文件不存在则为新股
                        # 用来判断当前数据长度
                        shape_data = pd.read_csv(InfoPath +
                                                 "StockInfo\\MoneyFlow\\" +
                                                 "000001" + ".csv")
                        Data_zeros = pd.DataFrame(
                            np.zeros((shape_data.shape[0] - 1,
                                      shape_data.shape[1])),
                            columns=[
                                'ts_code', 'trade_date', 'buy_sm_vol',
                                'buy_sm_amount', 'sell_sm_vol',
                                'sell_sm_amount', 'buy_md_vol',
                                'buy_md_amount', 'sell_md_vol',
                                'sell_md_amount', 'buy_lg_vol',
                                'buy_lg_amount', 'sell_lg_vol',
                                'sell_lg_amount', 'buy_elg_vol',
                                'buy_elg_amount', 'sell_elg_vol',
                                'sell_elg_amount', 'net_mf_vol',
                                'net_mf_amount'
                            ])
                        # 每日指标
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                Money_flow = pro.moneyflow(
                                    ts_code=Stock_code,
                                    start_date=Start_Date,
                                    end_date=End_Date)
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if Money_flow is None:
                                    continue
                                else:
                                    break
                        if Money_flow.empty:
                            pass
                        else:
                            Money_flow = pd.concat([Data_zeros, Money_flow],
                                                   axis=0)
                            Money_flow['ts_code'] = Stock_code
                            Money_flow.index = range(Money_flow.shape[0])
                            Money_flow['trade_date'] = shape_data['trade_date']
                            # 新股个股资金流量保留header
                            Money_flow.to_csv(
                                InfoPath + "StockInfo\\MoneyFlow\\" +
                                Stock_code[0:6] + ".csv",
                                index=False,
                                # header=False,
                                mode='a')
                    else:
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                Money_flow = pro.moneyflow(
                                    ts_code=Stock_code,
                                    start_date=Start_Date,
                                    end_date=End_Date)
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if Money_flow is None:
                                    continue
                                else:
                                    break
                        if Money_flow.empty:  # 当日更新有可能为空
                            Money_flow = pd.DataFrame(
                                [
                                    Stock_code, End_Date, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                ],
                                index=[
                                    'ts_code', 'trade_date', 'buy_sm_vol',
                                    'buy_sm_amount', 'sell_sm_vol',
                                    'sell_sm_amount', 'buy_md_vol',
                                    'buy_md_amount', 'sell_md_vol',
                                    'sell_md_amount', 'buy_lg_vol',
                                    'buy_lg_amount', 'sell_lg_vol',
                                    'sell_lg_amount', 'buy_elg_vol',
                                    'buy_elg_amount', 'sell_elg_vol',
                                    'sell_elg_amount', 'net_mf_vol',
                                    'net_mf_amount'
                                ]).T
                        else:
                            Money_flow = pd.merge(Money_flow,
                                                  trading_day,
                                                  on='trade_date',
                                                  how='outer',
                                                  sort=True)
                            # 其他信息向前填充
                            Money_flow['ts_code'] = Money_flow[
                                'ts_code'].fillna(method='ffill').fillna(
                                    method='bfill')
                            Money_flow = Money_flow.fillna(0)
                            # 个股资金流量直接追加模式
                        Money_flow.to_csv(InfoPath + "StockInfo\\MoneyFlow\\" +
                                          Stock_code[0:6] + ".csv",
                                          index=False,
                                          header=False,
                                          mode='a')
                except BaseException:
                    log.logger.error(("股票" + Stock_code + "-Day数据在" + Start_Date + "缺失,请检查!"))
        return

    def index_basic(self, trading_day, retry_count=50, pause=5):
        '''
        指数基础数据下载部分
        交易所信息，指数信息
        MSCI	MSCI指数    SSE	    上交所指数
        CSI	    中证指数    SZSE	深交所指数
        SW	    申万指数     OTH	其他指数
        CICC	中金所指数
        '''
        pro = self.pro
        log = self.log
        trading_day = pd.DataFrame(trading_day)
        trading_day.columns = ['trade_date']
        """指数基本信息"""
        index_company = ['MSCI', 'CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH']
        for i, index_code in enumerate(index_company):
            # print(index_code)
            for _ in range(retry_count):
                # 加入重试机制，防止访问过于频繁
                try:
                    index_basic = pro.index_basic(
                        market=index_code,
                        fields=
                        'ts_code,name,fullname,market,publisher,index_type,\
                        category,base_date,base_point,list_date,weight_rule,desc,exp_date'
                    )
                except BaseException:
                    time.sleep(pause)
                else:
                    break
            index_basic.to_csv(
                InfoPath + "IndexInfo\\BasicData\\" + index_code + ".csv",
                index=False,
                header=True,
                mode='w',
                encoding='GBK'
            )
        return

    def index_day(self, trading_day, retry_count=50, pause=5):
        '''
        指数日线数据下载部分
        指数成分与权重
        指数每日指标更新
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log
        trading_day = pd.DataFrame(trading_day)
        trading_day.columns = ['trade_date']

        # 日线行情更新指数
        # 依次上证综指、上证50、沪深300、深证成指、中小板、创业板
        # 中证100、中证200、中证500、中证800、中证1000
        index_common = [
            '000001.SH',
            '000016.SH',
            '399300.SZ',
            '399001.SZ',
            '399005.SZ',
            '399006.SZ',
            '000903.SH',
            '000904.SH',
            '000905.SH',
            '000906.SH',
            '000852.SH',
        ]
        # 每日指标更新指数
        index_common2 = [
            '000001.SH', '000005.SH', '000006.SH', '000016.SH', '000905.SH',
            '399001.SZ', '399005.SZ', '399006.SZ', '399016.SZ', '399300.SZ'
        ]
        try:
            for i, index_code in enumerate(index_common):
                """指数日线行情"""
                for _ in range(retry_count):
                    # 加入重试机制，防止访问过于频繁
                    try:
                        index_day = pro.index_daily(ts_code=index_code,
                                                    start_date=Start_Date,
                                                    end_date=End_Date)
                    except BaseException:
                        time.sleep(pause)
                    else:
                        break
                index_day = pd.merge(index_day,
                                     trading_day,
                                     on='trade_date',
                                     how='outer',
                                     sort=True)
                index_day = index_day.fillna(method='ffill').fillna(0)
                index_day['ts_code'] = index_code
                # 指数收益率统一改成小数模式
                index_day['ret'] = index_day['pct_chg'] / 100
                # 填充计算收益率产生的inf值
                index_day = index_day.replace([np.inf, -np.inf], [0, 0])
                index_day.to_csv(DataPath + "Day\\Index\\" + index_code[0:6] +
                                 '_' + index_code[7:9] + ".csv",
                                 index=False,
                                 header=False,
                                 mode='a')
                """指数成分与权重"""
                day_date = trading_day['trade_date'].tolist()
                for i, date in enumerate(day_date):
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            index_weight = pro.index_weight(
                                index_code=index_code,
                                start_date=date,
                                end_date=date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if index_weight.empty:
                        continue
                    else:
                        index_weight.to_csv(
                            InfoPath + "IndexInfo\\IndexWeight\\" +
                            index_code[0:6] + '_' + index_code[7:9] + "\\" +
                            date + ".csv",
                            index=False,
                            # header=False,
                            mode='w')
        except BaseException:
            log.logger.error(("指数" + index_code + "-(index_common)Day数据在" + Start_Date + "缺失,请检查!"))

        try:
            """大盘指数每日指标"""
            for i, index_code in enumerate(index_common2):
                for _ in range(retry_count):
                    # 加入重试机制，防止访问过于频繁
                    try:
                        index_dayquota = pro.index_dailybasic(
                            ts_code=index_code,
                            start_date=Start_Date,
                            end_date=End_Date,
                            fields=
                            'ts_code,trade_date,total_mv,float_mv, total_share,\
                            float_share,free_share,turnover_rate,turnover_rate_f,pe,pe_ttm,pb'
                        )
                    except BaseException:
                        time.sleep(pause)
                    else:
                        break
                if index_dayquota.empty:
                    log.logger.info(index_code + " in " + Start_Date + " is empty!")
                else:
                    index_dayquota = pd.merge(index_dayquota,
                                              trading_day,
                                              on='trade_date',
                                              how='outer',
                                              sort=True)
                    index_dayquota['ts_code'] = index_code
                    index_dayquota = index_dayquota.fillna(
                        method='ffill').fillna(0)
                    index_dayquota.to_csv(InfoPath + "IndexInfo\\IndexDayQuota\\" +
                                          index_code[0:6] + '_' + index_code[7:9] +
                                          ".csv",
                                          index=False,
                                          header=False,
                                          mode='a')
        except BaseException:
            log.logger.error(("指数" + index_code + "-(index_common2)Day数据在" + Start_Date + "缺失,请检查!"))
        return

    def future_basic(self, retry_count=50, pause=5):
        '''
        期货基础数据下载部分
        交易日历、合约信息
        CFFEX    中金所
        SHFE     上期所
        DCE      大商所
        CZCE     郑商所
        INE      能源中心
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log
        future_list = []  # 存储所有期货合约代码
        future_list2 = []
        fut_exchg = ['CFFEX', 'SHFE', 'DCE', 'CZCE', 'INE']
        """期货交易日历"""
        Tradying_day = pro.trade_cal(exchange='SHFE',
                                     start_date=Start_Date,
                                     end_date=End_Date,
                                     is_open='1')
        Tradying_day.to_csv(InfoPath + "FutureInfo\\BasicData\\TradingDay.csv",
                            index=False,
                            header=False,
                            mode='a')
        """期货合约信息"""
        for i, future_code in enumerate(fut_exchg):
            # 主力加普通合约
            for _ in range(retry_count):
                # 加入重试机制，防止访问过于频繁
                try:
                    future_common = pro.fut_basic(
                        exchange=future_code,
                        fut_type='',
                        fields='ts_code,symbol,exchange,name,fut_code,\
                            multiplier,trade_unit,per_unit,quote_unit,\
                            quote_unit_desc,d_mode_desc,list_date,delist_date\
                            ,d_month,last_ddate,trade_time_desc')
                except BaseException:
                    time.sleep(pause)
                else:
                    break
            if future_common.empty:
                pass
            else:
                future_common.to_csv(
                    InfoPath + "FutureInfo\\BasicData\\" + future_code +
                    "_Comm.csv",
                    index=False,
                    # header=False,
                    mode='w',encoding='GBK')
            future_list.extend(future_common['ts_code'])
            # 主力连续合约
            for _ in range(retry_count):
                # 加入重试机制，防止访问过于频繁
                try:
                    future_zhuli = pro.fut_basic(
                        exchange=future_code,
                        fut_type='2',
                        fields='ts_code,symbol,exchange,name,fut_code,\
                            multiplier,trade_unit,per_unit,quote_unit,\
                            quote_unit_desc,d_mode_desc,list_date,delist_date\
                            ,d_month,last_ddate,trade_time_desc')
                except BaseException:
                    time.sleep(pause)
                else:
                    break
            future_list2.extend(future_zhuli['ts_code'])
            if future_zhuli.empty:
                pass
            else:
                future_zhuli.to_csv(
                    InfoPath + "FutureInfo\\BasicData\\" + future_code +
                    "_ZhuLi.csv",
                    index=False,
                    # header=False,
                    mode='w',encoding='GBK')
        pd.DataFrame(future_list).to_csv(
            InfoPath + "FutureInfo\\BasicData\\" + "Future_List" + ".csv",
            index=False,
            # header=False,
            mode='w',encoding='GBK')
        return future_list,future_list2

    def future_day(self, trading_day, future_list, retry_count=50, pause=5):
        '''
        期货数据下载部分
        行情数据、合约信息
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log
        trading_day = pd.DataFrame(trading_day)
        trading_day.columns = ['trade_date']
        try:
            for i, future_code in enumerate(future_list):
                """期货行情信息"""
                # 判断是否是新合约
                if ((not os.path.exists(DataPath + "Day\\CFFEX\\" +
                                        future_code[:-4] + ".csv"))
                        and (not os.path.exists(DataPath + "Day\\SHFE\\" +
                                                future_code[:-4] + ".csv"))
                        and (not os.path.exists(DataPath + "Day\\DCE\\" +
                                                future_code[:-4] + ".csv"))
                        and (not os.path.exists(DataPath + "Day\\CZCE\\" +
                                                future_code[:-4] + ".csv"))
                        and (not os.path.exists(DataPath + "Day\\INE\\" +
                                                future_code[:-4] + ".csv"))):
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            Day_data = pro.fut_daily(ts_code=future_code,
                                                     start_date=Start_Date,
                                                     end_date=End_Date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if Day_data.empty:
                        pass
                    else:
                        Day_data = Day_data.sort_values(by='trade_date')
                        Day_data.index = range(Day_data.shape[0])
                        Day_data = Day_data.fillna(method='ffill').fillna(0)
                        if future_code[-3:] == "CFX":
                            Day_data.to_csv(
                                DataPath + "Day\\CFFEX\\" + future_code[:-4] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='a')
                        elif future_code[-3:] == "SHF":
                            Day_data.to_csv(
                                DataPath + "Day\\SHFE\\" + future_code[:-4] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='a')
                        elif future_code[-3:] == "DCE":
                            Day_data.to_csv(
                                DataPath + "Day\\DCE\\" + future_code[:-4] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='a')
                        elif future_code[-3:] == "ZCE":
                            Day_data.to_csv(
                                DataPath + "Day\\CZCE\\" + future_code[:-4] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='a')
                        elif future_code[-3:] == "INE":
                            Day_data.to_csv(
                                DataPath + "Day\\INE\\" + future_code[:-4] +
                                ".csv",
                                index=False,
                                # header=False,
                                mode='a')
                else:
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            Day_data = pro.fut_daily(ts_code=future_code,
                                                     start_date=Start_Date,
                                                     end_date=End_Date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if Day_data.empty:  # 判断合约是否到期
                        pass
                    else:
                        # print(future_code)
                        Day_data = Day_data.sort_values(by='trade_date')
                        Day_data = Day_data.fillna(method='ffill').fillna(0)
                        if future_code[-3:] == "CFX":
                            Day_data.to_csv(DataPath + "Day\\CFFEX\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                        elif future_code[-3:] == "SHF":
                            Day_data.to_csv(DataPath + "Day\\SHFE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                        elif future_code[-3:] == "DCE":
                            Day_data.to_csv(DataPath + "Day\\DCE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                        elif future_code[-3:] == "ZCE":
                            Day_data.to_csv(DataPath + "Day\\CZCE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                        elif future_code[-3:] == "INE":
                            Day_data.to_csv(DataPath + "Day\\INE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                """每日结算参数"""
                # 只有上期所、郑商所、国际能源交易中心有结算数据
                if not os.path.exists(InfoPath + "FutureInfo\\FutSettle\\" +
                                      future_code[:-4] + ".csv"):
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            settle_day = pro.fut_settle(
                                ts_code=future_code,
                                # trade_date='20181114',
                                start_date=Start_Date,
                                end_date=End_Date,
                                fields=
                                'ts_code,trade_date,settle,trading_fee_rate,trading_fee,\
                                delivery_fee,b_hedging_margin_rate,s_hedging_margin_rate,\
                                long_margin_rate,short_margin_rate,offset_today_fee,exchange'

                                # ,exchange='SHFE'
                            )
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if settle_day.empty:
                        pass
                    else:
                        settle_day = settle_day.sort_values(by='trade_date')
                        settle_day = settle_day.fillna(
                            method='ffill').fillna(0)
                        settle_day['ts_code'] = future_code
                        settle_day.to_csv(InfoPath +
                                          "FutureInfo\\FutSettle\\" +
                                          future_code[:-4] + ".csv",
                                          index=False,
                                          header=True,
                                          mode='a')
                else:
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            settle_day = pro.fut_settle(
                                ts_code=future_code,
                                start_date=Start_Date,
                                end_date=End_Date,
                                fields=
                                'ts_code,trade_date,settle,trading_fee_rate,trading_fee,\
                                    delivery_fee,b_hedging_margin_rate,s_hedging_margin_rate,\
                                    long_margin_rate,short_margin_rate,offset_today_fee,exchange'
                            )
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if settle_day.empty:  # 判断合约是否到期
                        pass
                        # print(future_code)
                    else:
                        settle_day = settle_day.sort_values(by='trade_date')
                        settle_day = settle_day.fillna(
                            method='ffill').fillna(0)
                        settle_day['ts_code'] = future_code
                        settle_day.to_csv(InfoPath +
                                          "FutureInfo\\FutSettle\\" +
                                          future_code[:-4] + ".csv",
                                          index=False,
                                          header=False,
                                          mode='a')
        except BaseException:
            log.logger.error(
                ("期货" + future_code + "-Day数据在" + Start_Date + "缺失,请检查!"))

        try:
            """南华期货指数"""
            index_list = [
                'NHAI.NH', 'NHCI.NH', 'NHECI.NH', 'NHFI.NH', 'NHII.NH',
                'NHMI.NH', 'NHNFI.NH', 'NHPMI.NH', 'A.NH', 'AG.NH', 'AL.NH',
                'AP.NH', 'AU.NH', 'BB.NH', 'BU.NH', 'C.NH', 'CF.NH', 'CS.NH',
                'CU.NH', 'CY.NH', 'ER.NH', 'FB.NH', 'FG.NH', 'FU.NH', 'HC.NH',
                'I.NH', 'J.NH', 'JD.NH', 'JM.NH', 'JR.NH', 'L.NH', 'LR.NH',
                'M.NH', 'ME.NH', 'NI.NH', 'P.NH', 'PB.NH', 'PP.NH', 'RB.NH',
                'RM.NH', 'RO.NH', 'RS.NH', 'RU.NH', 'SC.NH', 'SF.NH', 'SM.NH',
                'SN.NH', 'SP.NH', 'SR.NH', 'TA.NH', 'TC.NH', 'V.NH', 'WR.NH',
                'WS.NH', 'Y.NH', 'ZN.NH'
            ]
            for i, index_code in enumerate(index_list):
                if not os.path.exists(InfoPath + "FutureInfo\\FutIndex\\" +
                                      index_code[:-3] + "_NH.csv"):
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            index_day = pro.index_daily(ts_code=index_code,
                                                        start_date=Start_Date,
                                                        end_date=End_Date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if index_day.empty:
                        pass
                    else:
                        index_day = index_day.sort_values(by='trade_date')
                        index_day = index_day.fillna(method='ffill').fillna(0)
                        index_day['ts_code'] = index_code
                        index_day.to_csv(InfoPath + "FutureInfo\\FutIndex\\" +
                                         index_code[:-3] + "_NH.csv",
                                         index=False,
                                         header=True,
                                         mode='a')
                else:
                    for _ in range(retry_count):
                        # 加入重试机制，防止访问过于频繁
                        try:
                            index_day = pro.index_daily(ts_code=index_code,
                                                        start_date=Start_Date,
                                                        end_date=End_Date)
                        except BaseException:
                            time.sleep(pause)
                        else:
                            break
                    if index_day.empty:
                        pass
                    else:
                        index_day = index_day.sort_values(by='trade_date')
                        index_day = index_day.fillna(method='ffill').fillna(0)
                        index_day.to_csv(InfoPath + "FutureInfo\\FutIndex\\" +
                                         index_code[:-3] + "_NH.csv",
                                         index=False,
                                         header=False,
                                         mode='a')
        except BaseException:
            log.logger.error("期货指数" + index_code + "-Day数据在" + Start_Date + "缺失,请检查!")

    def option_basic(self, retry_count=100, pause=2):
        '''
        四大交易所期权数据
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        opt_exchg = ['CZCE', 'SHFE', 'DCE', 'SSE']
        # 获取期权合约信息
        option_list = []
        for i, exchg_code in enumerate(opt_exchg):
            for _ in range(retry_count):
                # 加入重试机制，防止访问过于频繁
                try:
                    option_data = pro.opt_basic(exchange=exchg_code)
                except BaseException:
                    time.sleep(pause)
                else:
                    if option_data is None:
                        continue
                    else:
                        break
            option_list.extend(option_data['ts_code'])
            option_data.to_csv(InfoPath + "OptionInfo\\BasicData\\" +
                               exchg_code[0:6] + ".csv",
                               index=False,
                               mode='w', encoding='GBK')
        # 获取交易日
        Trading_Day = pro.trade_cal(exchange='SSE',
                                    start_date=Start_Date,
                                    end_date=End_Date,
                                    is_open=1)
        trading_day = Trading_Day['cal_date'].tolist()
        return (option_list, trading_day)

    def option_day(self, trading_day, option_list, retry_count=100, pause=2):
        '''
        股票分钟行情数据下载部分
        tushare分钟数据一次限定7000行
        更新模式：股票、指数按照天存储
                 期货按照合约存储
        '''
        Start_Date = self.Start_Date
        End_Date = self.End_Date
        pro = self.pro
        log = self.log

        for i, option_code in enumerate(option_list):
            """期权日线数据"""
            try:
                for _ in range(retry_count):
                    # 加入重试机制，防止访问过于频繁
                    try:
                        option_day = pro.opt_daily(ts_code=option_code,
                                                   start_date=Start_Date,
                                                   end_date=End_Date)
                    except BaseException:
                        time.sleep(pause)
                    else:
                        if option_day is None:
                            continue
                        else:
                            break
                if option_day.empty:  # 退市或者停牌
                    pass
                    # log.logger.info(option_code + "-Day数据在" + Start_Date + "缺失,请检查!")
                else:
                    option_day = option_day.sort_values(by='trade_date')
                    option_day = option_day.fillna(method="ffill").fillna(0)
                    if option_code[-2:] == 'SH':
                        # 保存数据
                        if not os.path.exists(DataPath + "Day\\OPT\\SSE\\" +
                                              option_code[:-3] + "_" +
                                              option_code[-2:] + ".csv"):
                            option_day.to_csv(DataPath + "Day\\OPT\\SSE\\" +
                                              option_code[:-3] + "_" +
                                              option_code[-2:] + ".csv",
                                              index=False,
                                              mode='a')
                        else:
                            option_day.to_csv(DataPath + "Day\\OPT\\SSE\\" +
                                              option_code[:-3] + "_" +
                                              option_code[-2:] + ".csv",
                                              index=False,
                                              header=False,
                                              mode='a')
                    elif option_code[-3:] == 'SHF':
                        # 保存数据
                        if not os.path.exists(DataPath + "Day\\OPT\\SHFE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv"):
                            option_day.to_csv(DataPath + "Day\\OPT\\SHFE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv",
                                              index=False,
                                              mode='a')
                        else:
                            option_day.to_csv(DataPath + "Day\\OPT\\SHFE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv",
                                              index=False,
                                              header=False,
                                              mode='a')
                    elif option_code[-3:] == 'DCE':
                        # 保存数据
                        if not os.path.exists(DataPath + "Day\\OPT\\DCE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv"):
                            option_day.to_csv(DataPath + "Day\\OPT\\DCE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv",
                                              index=False,
                                              mode='a')
                        else:
                            option_day.to_csv(DataPath + "Day\\OPT\\DCE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv",
                                              index=False,
                                              header=False,
                                              mode='a')
                    else:
                        # 保存数据
                        if not os.path.exists(DataPath + "Day\\OPT\\CZCE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv"):
                            option_day.to_csv(DataPath + "Day\\OPT\\CZCE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv",
                                              index=False,
                                              mode='a')
                        else:
                            option_day.to_csv(DataPath + "Day\\OPT\\CZCE\\" +
                                              option_code[:-4] + "_" +
                                              option_code[-3:] + ".csv",
                                              index=False,
                                              header=False,
                                              mode='a')
            except Exception:
                log.logger.error(option_code + " meets a serious error and needs to fill")
        return


def day_func(params):
    pro = ts.pro_api()
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 设置日志文件
    log = Logger(Pre_path + "\\Logging\\T_DataUpdate_Day\\" + 'Day_Error.log',
                 level='info')
    start_date = params['start_date']
    end_date = params['end_date']
    trading_day = params['trading_day']
    data_update = Data_Update(start_date, end_date, pro, log)
    if params['func_name'] == 'stock_day':
        stock_list = params['stock_list']
        data_update.stock_day(trading_day, stock_list)
    elif params['func_name'] == 'future_day' :
        future_list = params['future_list']
        data_update.future_day(trading_day, future_list)
    else:
        option_list = params['option_list']
        data_update.option_day(trading_day, option_list)
    return


def set_params(code_list, trading_day, start_date, end_date, func_name):
    td = {
        'trading_day': trading_day,
        'start_date': start_date,
        'end_date': end_date,
        'func_name': func_name
    }
    params = []
    if func_name == 'stock_day':
        for i, sec_code in enumerate(code_list):
            td['stock_list'] = sec_code
            params.append(td.copy())
    elif func_name == 'future_day':
        for i, sec_code in enumerate(code_list):
            td['future_list'] = sec_code
            params.append(td.copy())
    else:
        for i, sec_code in enumerate(code_list):
            td['option_list'] = sec_code
            params.append(td.copy())
    return params


if __name__ == "__main__":
    """初始化tushare接口"""
    pro = ts.pro_api()
    """设置日志文件"""
    log = Logger(Pre_path + "\\Logging\\T_DataUpdate_Day\\" + 'Day_Error.log',level='info')
    """建立数据连接"""
    # db = MySQLConn.DBCONN()
    """判断当前日期是否已经更新"""
    TradeDate = pd.read_csv(Pre_path+"\\TradeDate.csv")
    trading_day = [str(x) for x in TradeDate['trade_date']]
    Start_Date = str(trading_day[0])
    End_Date = str(trading_day[-1])
    print("Data of today begin to update !")
    # =========================================================
    # =========================================================
    """更新股票"""
    data_update = Data_Update(Start_Date, End_Date, pro, log)
    stock_list = data_update.stock_basic()
    print("The update of stock_basic is finished~")
    exit_list = [x[0:6] + ".SZ" for x in os.listdir(DataPath + "\\Day\\SZE\\")] \
                + [i[0:6] + ".SH" for i in os.listdir(DataPath + "\\Day\\SSE\\")]
    stock_list_o = list(set(exit_list).union(set(stock_list.tolist())))
    stock_list_o.sort(key=lambda x: int(x[0:6]))
    stock_list = [stock_list_o[i:i + 50] for i in range(0, len(stock_list_o), 50)]
    para_x = set_params(stock_list, trading_day, Start_Date, End_Date,"stock_day")
    pool = multiprocessing.Pool(16)
    # 多进程map中不能有传入地址参数pro、log
    pool.map(day_func, para_x)
    pool.close()
    pool.join()
    print("The update of stock_day is finished~")
    # =========================================================
    # =========================================================
    """更新指数"""
    data_update.index_basic(trading_day)
    print("The update of index_basic is finished~")
    data_update.index_day(trading_day)
    print("The update of index_day is finished~")
    # =========================================================
    # =========================================================
    """更新期货"""
    # 获取当前期货列表
    future_list, future_list2 = data_update.future_basic()
    print("The update of future_basic is finished~")
    year_list = ['19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
    year_list2 = ['1901', '1902', '1903', '1904','1905','1906']
    future_list = [x for x in future_list if re.sub("\D", "", x)[0:2] in year_list and (
                   re.sub("\D", "", x)[0:4] not in year_list2)] + future_list2
    future_list = [future_list[i:i + 50] for i in range(0, len(future_list), 50)]
    para_y = set_params(future_list, trading_day, Start_Date, End_Date,"future_day")
    pool = multiprocessing.Pool(8)
    pool.map(day_func, para_y)
    pool.close()
    pool.join()
    print("The update of future_day is finished~")
    # =========================================================
    # =========================================================
    """期货指数去重"""
    file_list = [x for x in os.listdir(InfoPath + "\\FutureInfo\\FutIndex\\")]
    os.mkdir(InfoPath + "\\FutureInfo\\temp")
    for i, file_name in enumerate(file_list):
        data = pd.read_csv(InfoPath + "\\FutureInfo\\FutIndex\\" + file_name)
        data = data.drop_duplicates(['ts_code','trade_date'])
        data.to_csv(InfoPath + "FutureInfo\\temp\\" + file_name,index=False,mode='a')
        shutil.copy(InfoPath + "\\FutureInfo\\temp\\" + file_name, InfoPath + "\\FutureInfo\\FutIndex\\")
    shutil.rmtree(InfoPath + "\\FutureInfo\\temp")
    print("The drop_duplicated of future_index is finish!")
    # =========================================================
    # =========================================================
    """更新期权数据"""
    option_list, trading_day = data_update.option_basic()
    print("The update of option_basic is finished~")
    option_list = [option_list[i:i + 50] for i in range(0, len(option_list), 50)]
    para_z = set_params(option_list, trading_day, Start_Date, End_Date, "option_day")
    pool = multiprocessing.Pool(16)
    pool.map(day_func, para_z)
    pool.close()
    pool.join()
    print("The update of option_day is finished~")
    # =========================================================
    # =========================================================
    """自动检查是否有缺失日线数据"""
    tradingday = pro.trade_cal(exchange='', start_date='19951231',
                  end_date=End_Date, is_open=1).cal_date.tolist()
    files_list = os.listdir(InfoPath + "StockInfo\\DayQuota\\")
    for stock_name in files_list:
        if stock_name[0] != str(6):
            stock_day = pd.read_csv(DataPath + "Day\\SZE\\" + stock_name)
        else:
            stock_day = pd.read_csv(DataPath + "Day\\SSE\\" + stock_name)
        if stock_day.shape[0] < len(tradingday): # 缺失数据
            stock_code = [stock_name[0:6]+".SH" if stock_name[0]=="6" else stock_name[0:6]+".SZ"]
            trade_date = [x for x in tradingday if x not in str(stock_day["trade_date"].tolist())]
            data_update = Data_Update(trade_date[0], trade_date[-1], pro, log)
            data_update.stock_day(tradingday[tradingday.index(trade_date[0]):tradingday.index(trade_date[-1])+1],stock_code)
        elif stock_day.shape[0] > len(tradingday):
            # 行情数据去重
            stock_day = stock_day.drop_duplicates(subset=["trade_date"])
            stock_day["trade_date"] = stock_day["trade_date"].astype('int64')
            stock_day = stock_day.sort_values(by="trade_date")
            if stock_name[0] != str(6):
                stock_day.to_csv(DataPath + "Day\\SZE\\" + stock_name, index=False, mode='w')
            else:
                stock_day.to_csv(DataPath + "Day\\SSE\\" + stock_name, index=False, mode='w')
            # 每日指标去重
            print(stock_name)
            quato_data = pd.read_csv(InfoPath + "StockInfo\\DayQuota\\" + stock_name)
            quato_data = quato_data.drop_duplicates(subset=["trade_date"])
            quato_data["trade_date"] = quato_data["trade_date"].astype('int64')
            quato_data = quato_data.sort_values(by="trade_date")
            quato_data.to_csv(InfoPath + "StockInfo\\DayQuota\\" + stock_name, index=False, mode='w')
    print("The data of lost data filled down~")

