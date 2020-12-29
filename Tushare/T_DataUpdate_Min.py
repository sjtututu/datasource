#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/8 16:17
# @Author: Tu Xinglong
# @File  : T_DataUpdate_Min.py

import os
import re
import sys
import time
import warnings

import pandas as pd
import tushare as ts
import datetime as dt
import multiprocessing
from dateutil.parser import parse

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.abspath(os.path.dirname(CurrentPath))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
DataPath = Pre_path + '\\MarketData\\'

from LoggingPlus import Logger


class Data_Update(object):
    '''
    股票行情信息数据获取类
    tushare接口调取数据需要注意的问题：
    1.交易日里与行情数据的日期顺序相反
    2.日线一次性调取数据的量不能超过4000条
    3.需要高级权限才能调取分钟K线及指数权重
    4.返回的数据剔除了停牌日，需自行补上填充
    5.需要付费才能进入tushare高级用户群
    6.每天更新数据之前判断是否有新股
    7.接口返回None时请更新tushare
    8.分钟线一次性调取的数据不超过7000条
    9.股指期货交易时间9:15开始，15:15结束
    10.商品期货时间有夜盘，一般21:00-23:00
    或者持续到1:00,原油期货持续到2:30,这样在下载
    1min线数据的时候不可切割为28天，这里特别需要注意
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

    def basic_data(self, retry_count=50, pause=5):
        '''
        股票交易日，期货交易日
        股票列表，期货合约列表
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

        # 指数列表
        index_list = [
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

        # 获取期货交易日
        F_tradyingday = pro.trade_cal(exchange='SHFE',
                                      start_date=Start_Date,
                                      end_date=End_Date,
                                      is_open='1')
        future_day = F_tradyingday['cal_date'].tolist()

        # 获取期货合约列表
        future_list = []
        fut_exchg = ['CFFEX', 'DCE', 'SHFE', 'CZCE', 'INE']  #
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
            future_list.extend(future_common['ts_code'])
        # future_exist = os.listdir(DataPath + "Minute\\DCE\\")
        # future_exist_new = [x[:-4] for x in future_exist]
        year_list = [
            '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'
        ]
        year_list2 = ['1901', '1902', '1903', '1904','1905','1906']
        future_list = [
            x for x in future_list
            if re.sub("\D", "", x)[0:2] in year_list and (
                re.sub("\D", "", x)[0:4] not in year_list2)
        ]

        return (stock_day, stock_list, index_list, future_day, future_list)

    def stock_min(self, stock_day, stock_list, retry_count=200, pause=5):
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

        if (False):  # 首次更新
            try:
                for m, stock_code in enumerate(stock_list):
                    """股票分钟数据"""
                    # 对交易日进行等分切割
                    day_avg28 = [  # 按照28天切割成一个小list
                        stock_day[i:i + 28]
                        for i in range(0, len(stock_day), 28)
                    ]
                    for i, trade_date in enumerate(day_avg28):
                        if i != len(day_avg28) - 1:
                            startdate = trade_date[0]
                            enddate = day_avg28[i + 1][0]
                        else:
                            startdate = trade_date[0]
                            enddate = trade_date[-1]
                        if ((os.path.isdir(DataPath + "Minute\\SSE\\" +
                                           stock_code[0:6]))
                                or (os.path.isdir(DataPath + "Minute\\SZE\\" +
                                                  stock_code[0:6]))):  # 无新股产生
                            pass
                        else:
                            if stock_code[7:9] == "SZ":  # 创建新股文件夹
                                os.mkdir(DataPath + "Minute\\SZE\\" +
                                         stock_code[0:6])
                            else:
                                os.mkdir(DataPath + "Minute\\SSE\\" +
                                         stock_code[0:6])
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                stock_min = ts.pro_bar(
                                    ts_code=stock_code,
                                    api=pro,
                                    start_date=startdate,  # 当前交易日
                                    end_date=enddate,  # 后一交易日
                                    freq='1min',
                                    asset='E',
                                    adj=None,
                                    adjfactor=False)  # 与复权数据配合使用
                                # False表示保留adjfactors
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if stock_min is None:
                                    continue
                                else:
                                    break
                        stock_min = pd.DataFrame(stock_min)
                        if stock_min.empty:
                            log.logger.info("股票" + stock_code + "-1min数据在" +
                                            startdate + "_" + enddate +
                                            "缺失,请检查!")
                        else:  # 首次保存与追加的区别
                            stock_min = stock_min.sort_values(by='trade_time')
                            stock_min = stock_min.fillna(
                                method="ffill").fillna(0)
                            if (not os.path.exists(  # 判断是否有新股
                                (DataPath + "Minute\\SZE\\" + stock_code[0:6] +
                                 "\\" + stock_code[0:6] + ".csv"))) and (
                                     not os.path.exists(
                                         (DataPath + "Minute\\SSE\\" +
                                          stock_code[0:6] + "\\" +
                                          stock_code[0:6] + ".csv"))):
                                # 保存数据
                                if stock_code[7:9] == "SZ":
                                    stock_min.to_csv(
                                        DataPath + "Minute\\SZE\\" +
                                        stock_code[0:6] + "\\" +
                                        stock_code[0:6] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                                else:
                                    stock_min.to_csv(
                                        DataPath + "Minute\\SSE\\" +
                                        stock_code[0:6] + "\\" +
                                        stock_code[0:6] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                            else:
                                if stock_code[7:9] == "SZ":
                                    stock_min.to_csv(DataPath +
                                                     "Minute\\SZE\\" +
                                                     stock_code[0:6] + "\\" +
                                                     stock_code[0:6] + ".csv",
                                                     index=False,
                                                     header=False,
                                                     mode='a')
                                else:
                                    stock_min.to_csv(DataPath +
                                                     "Minute\\SSE\\" +
                                                     stock_code[0:6] + "\\" +
                                                     stock_code[0:6] + ".csv",
                                                     index=False,
                                                     header=False,
                                                     mode='a')
            except Exception:
                log.logger.error("Exception Logged " + stock_code)

        # ====================================================================
        # ====================================================================

        else:  # 增量更新
            for i, trade_date in enumerate(stock_day):  # 考虑到有可能不止更新一天
                if i != len(stock_day) - 1:
                    Start_Date = stock_day[i]
                    End_Date = stock_day[i + 1]
                    """股票分钟数据"""
                    try:
                        for m, stock_code in enumerate(stock_list):
                            if ((os.path.isdir(DataPath + "Minute\\SSE\\" +
                                               stock_code[0:6])) or
                                (os.path.isdir(DataPath + "Minute\\SZE\\" +
                                               stock_code[0:6]))):  # 无新股产生
                                pass
                            else:
                                if stock_code[7:9] == "SZ":  # 创建新股文件夹
                                    os.mkdir(DataPath + "Minute\\SZE\\" +
                                             stock_code[0:6])
                                else:
                                    os.mkdir(DataPath + "Minute\\SSE\\" +
                                             stock_code[0:6])
                            for _ in range(retry_count):
                                # 加入重试机制，防止访问过于频繁
                                try:
                                    stock_min = ts.pro_bar(
                                        ts_code=stock_code,
                                        api=pro,
                                        start_date=Start_Date,  # 当前交易日
                                        end_date=End_Date,  # 后一交易日
                                        freq='1min',
                                        asset='E',
                                        adj=None,
                                        adjfactor=False)  # 与复权数据配合使用
                                except BaseException:
                                    time.sleep(pause)
                                else:
                                    if stock_min is None:
                                        continue
                                    else:
                                        break
                            stock_min = pd.DataFrame(stock_min)
                            if stock_min.empty:  # 退市或者停牌
                                log.logger.info("股票" + stock_code +
                                                "-1min数据在" + Start_Date +
                                                "缺失,请检查!")
                            else:
                                stock_min = stock_min.sort_values(
                                    by='trade_time')
                                stock_min = stock_min.fillna(
                                    method="ffill").fillna(0)
                                # 保存数据
                                if stock_code[7:9] == "SZ":
                                    stock_min.to_csv(
                                        DataPath + "Minute\\SZE\\" +
                                        stock_code[0:6] + "\\" + Start_Date +
                                        ".csv",
                                        index=False,
                                        mode='w')
                                else:
                                    stock_min.to_csv(
                                        DataPath + "Minute\\SSE\\" +
                                        stock_code[0:6] + "\\" + Start_Date +
                                        ".csv",
                                        index=False,
                                        mode='w')
                                # stock_min.to_csv(
                                #     DataPath + "Minute\\temp\\" +
                                #     stock_code + "_" + Start_Date +
                                #     ".csv",
                                #     index=False,
                                #     mode='w')
                    except Exception:
                        log.logger.error("Exception Logged " + stock_code)
                else:
                    pass
        return

    def index_min(self, stock_day, index_list, retry_count=200, pause=2):
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

        if (False):  # 首次更新
            """指数分钟数据"""
            try:
                for n, index_code in enumerate(index_list):
                    # 对交易日进行等分切割
                    day_avg28 = [  # 按照28天切割成一个小list
                        stock_day[i:i + 28]
                        for i in range(0, len(stock_day), 28)
                    ]
                    for i, trade_date in enumerate(day_avg28):
                        if i != len(day_avg28) - 1:
                            startdate = trade_date[0]
                            enddate = day_avg28[i + 1][0]
                        else:
                            startdate = trade_date[0]
                            enddate = trade_date[-1]
                        if os.path.isdir(DataPath + "Minute\\INDEX\\" +
                                         index_code[0:6] + "_" +
                                         index_code[7:9]):
                            pass  # 判断是否加入新的指数
                        else:
                            os.mkdir(DataPath + "Minute\\INDEX\\" +
                                     index_code[0:6] + "_" + index_code[7:9])
                        for _ in range(retry_count):
                            # 加入重试机制，防止访问过于频繁
                            try:
                                index_min = ts.pro_bar(
                                    ts_code=index_code,
                                    api=pro,
                                    start_date=startdate,
                                    end_date=enddate,
                                    freq='1min',
                                    asset='I',
                                    adj=None)  # ma=[5, 10, 20]
                            except BaseException:
                                time.sleep(pause)
                            else:
                                if index_min is None:
                                    continue
                                else:
                                    break
                        index_min = pd.DataFrame(index_min)
                        if index_min.empty:
                            log.logger.info("指数" + index_code + "-1min数据在" +
                                            startdate + "_" + enddate +
                                            "缺失,请检查!")
                        else:
                            index_min = index_min.sort_values(by='trade_time')
                            index_min = index_min.fillna(
                                method="ffill").fillna(0)
                            if not os.path.exists(  # 判断是否有新股
                                (DataPath + "Minute\\INDEX\\" +
                                 index_code[0:6] + "_" + index_code[7:9] +
                                 "\\" + index_code[0:6] + ".csv")):
                                index_min.to_csv(
                                    DataPath + "Minute\\INDEX\\" +
                                    index_code[0:6] + "_" + index_code[7:9] +
                                    "\\" + index_code[0:6] + ".csv",
                                    index=False,
                                    # header=False,
                                    mode='a')
                            else:
                                index_min.to_csv(DataPath + "Minute\\INDEX\\" +
                                                 index_code[0:6] + "_" +
                                                 index_code[7:9] + "\\" +
                                                 index_code[0:6] + ".csv",
                                                 index=False,
                                                 header=False,
                                                 mode='a')
            except Exception:
                log.logger.error("Exception Logged " + index_code)
        # ====================================================================
        # ====================================================================
        else:  # 增量更新
            for i, trade_date in enumerate(stock_day):  # 考虑到有可能不止更新一天
                if i != len(stock_day) - 1:
                    Start_Date = stock_day[i]
                    End_Date = stock_day[i + 1]
                    """指数分钟数据"""
                    try:
                        for n, index_code in enumerate(index_list):
                            if os.path.isdir(DataPath + "Minute\\INDEX\\" +
                                             index_code[0:6] + "_" +
                                             index_code[7:9]):
                                pass  # 判断是否加入新的指数
                            else:
                                os.mkdir(DataPath + "Minute\\INDEX\\" +
                                         index_code[0:6] + "_" +
                                         index_code[7:9])
                            for _ in range(retry_count):
                                # 加入重试机制，防止访问过于频繁
                                try:
                                    index_min = ts.pro_bar(
                                        ts_code=index_code,
                                        api=pro,
                                        start_date=Start_Date,
                                        end_date=End_Date,
                                        freq='1min',
                                        asset='I',
                                        adj=None)
                                except BaseException:
                                    time.sleep(pause)
                                else:
                                    if index_min is None:
                                        continue
                                    else:
                                        break
                            index_min = pd.DataFrame(index_min)
                            if index_min.empty:
                                log.logger.info("指数" + index_code +
                                                "-1min数据在" + Start_Date +
                                                "缺失,请检查!")
                            else:
                                index_min = index_min.sort_values(
                                    by='trade_time')
                                index_min = index_min.fillna(
                                    method="ffill").fillna(0)
                                index_min.to_csv(DataPath + "Minute\\INDEX\\" +
                                                 index_code[0:6] + "_" +
                                                 index_code[7:9] + "\\" +
                                                 Start_Date + ".csv",
                                                 index=False,
                                                 mode='w')
                    except Exception:
                        log.logger.error("Exception Logged " + index_code)
                else:
                    pass
        return

    def future_min(self, future_day, future_list, retry_count=200, pause=2):
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

        if (False):  # 首次更新
            """期货分钟数据"""
            try:
                for s, future_code in enumerate(future_list):
                    # 对交易日进行等分切割
                    # 期货不同于股票和指数
                    # 交易时间更长，周期取12
                    # 节假日之后首个交易日9点开始
                    # 保存之前需删除最后一个重复值
                    day_avg12 = [  # 按照12天切割成一个小list
                        future_day[i:i + 12]
                        for i in range(0, len(future_day), 12)
                    ]
                    for i, trade_date in enumerate(day_avg12):
                        if i != len(day_avg12) - 1:
                            startdate = trade_date[0]
                            enddate = day_avg12[i + 1][0]
                        else:
                            startdate = trade_date[0]
                            enddate = trade_date[-1]
                        if ((not os.path.exists(DataPath + "Minute\\CFFEX\\" +
                                                future_code[:-4] + ".csv")) and
                            (not os.path.exists(DataPath + "Minute\\SHFE\\" +
                                                future_code[:-4] + ".csv")) and
                            (not os.path.exists(DataPath + "Minute\\DCE\\" +
                                                future_code[:-4] + ".csv")) and
                            (not os.path.exists(DataPath + "Minute\\CZCE\\" +
                                                future_code[:-4] + ".csv")) and
                            (not os.path.exists(DataPath + "Minute\\INE\\" +
                                                future_code[:-4] + ".csv"))):
                            for _ in range(retry_count):
                                # 加入重试机制，防止访问过于频繁
                                try:
                                    future_min = ts.pro_bar(
                                        api=pro,
                                        ts_code=future_code,
                                        asset='FT',
                                        adj=None,
                                        start_date=startdate,
                                        end_date=enddate,
                                        freq='1min')
                                except BaseException:
                                    time.sleep(pause)
                                else:
                                    if future_min is None:
                                        continue
                                    else:
                                        break
                            future_min = pd.DataFrame(future_min)
                            if future_min.empty:
                                log.logger.info("期货" + future_code +
                                                "-1min数据在" + startdate + "_" +
                                                enddate + "缺失,请检查!")
                            else:  # 首次出现的合约需要有header
                                future_min = future_min.sort_values(
                                    by='trade_time')
                                future_min = future_min.fillna(
                                    method='ffill').fillna(0)
                                future_min.index = range(future_min.shape[0])
                                if future_min['trade_time'].iloc[-1][
                                        -8:] == '00:00:00':
                                    future_min = future_min.drop(
                                        (future_min.shape[0] - 1))
                                else:
                                    pass
                                if future_code[-3:] == "CFX":
                                    future_min.to_csv(
                                        DataPath + "Minute\\CFFEX\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                                elif future_code[-3:] == "SHF":
                                    future_min.to_csv(
                                        DataPath + "Minute\\SHFE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                                elif future_code[-3:] == "DCE":
                                    future_min.to_csv(
                                        DataPath + "Minute\\DCE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                                elif future_code[-3:] == "ZCE":
                                    future_min.to_csv(
                                        DataPath + "Minute\\CZCE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                                elif future_code[-3:] == "INE":
                                    future_min.to_csv(
                                        DataPath + "Minute\\INE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        # header=False,
                                        mode='a')
                        else:
                            for _ in range(retry_count):
                                # 加入重试机制，防止访问过于频繁
                                try:
                                    future_min = ts.pro_bar(
                                        api=pro,
                                        ts_code=future_code,
                                        asset='FT',
                                        adj=None,
                                        start_date=startdate,
                                        end_date=enddate,
                                        freq='1min')
                                except BaseException:
                                    time.sleep(pause)
                                else:
                                    if future_min is None:
                                        continue
                                    else:
                                        break
                            future_min = pd.DataFrame(future_min)
                            if future_min.empty:  # 合约是否过期
                                log.logger.info("期货" + future_code +
                                                "-1min数据在" + startdate + "_" +
                                                enddate + "缺失,请检查!")
                            else:  # 首次出现的合约需要有header
                                future_min = future_min.sort_values(
                                    by='trade_time')
                                future_min = future_min.fillna(
                                    method='ffill').fillna(0)
                                future_min.index = range(future_min.shape[0])
                                if future_min['trade_time'].iloc[-1][
                                        -8:] == '00:00:00':
                                    future_min = future_min.drop(
                                        (future_min.shape[0] - 1))
                                else:
                                    pass
                                if future_code[-3:] == "CFX":
                                    future_min.to_csv(
                                        DataPath + "Minute\\CFFEX\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        header=False,
                                        mode='a')
                                elif future_code[-3:] == "SHF":
                                    future_min.to_csv(
                                        DataPath + "Minute\\SHFE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        header=False,
                                        mode='a')
                                elif future_code[-3:] == "DCE":
                                    future_min.to_csv(
                                        DataPath + "Minute\\DCE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        header=False,
                                        mode='a')
                                elif future_code[-3:] == "ZCE":
                                    future_min.to_csv(
                                        DataPath + "Minute\\CZCE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        header=False,
                                        mode='a')
                                elif future_code[-3:] == "INE":
                                    future_min.to_csv(
                                        DataPath + "Minute\\INE\\" +
                                        future_code[:-4] + ".csv",
                                        index=False,
                                        header=False,
                                        mode='a')
            except Exception:
                log.logger.error("Exception Logged " + future_code)
        # ====================================================================
        # ====================================================================
        else:  # 增量更新
            for i, trade_date in enumerate(future_day):  # 考虑到有可能不止更新一天
                if i != (len(future_day) - 1):
                    Start_Date = future_day[i]
                    End_Date = future_day[i + 1]
                    """期货分钟数据"""
                    try:
                        for s, future_code in enumerate(future_list):
                            if ((not os.path.exists(DataPath +
                                                    "Minute\\CFFEX\\" +
                                                    future_code[:-4] + ".csv"))
                                    and
                                (not os.path.exists(DataPath +
                                                    "Minute\\SHFE\\" +
                                                    future_code[:-4] + ".csv"))
                                    and
                                (not os.path.exists(DataPath +
                                                    "Minute\\DCE\\" +
                                                    future_code[:-4] + ".csv"))
                                    and
                                (not os.path.exists(DataPath +
                                                    "Minute\\CZCE\\" +
                                                    future_code[:-4] + ".csv"))
                                    and (not os.path.exists(DataPath +
                                                            "Minute\\INE\\" +
                                                            future_code[:-4] +
                                                            ".csv"))):
                                for _ in range(retry_count):
                                    # 加入重试机制，防止访问过于频繁
                                    try:
                                        future_min = ts.pro_bar(
                                            api=pro,
                                            ts_code=future_code,
                                            asset='FT',
                                            adj=None,
                                            start_date=Start_Date,
                                            end_date=End_Date,
                                            freq='1min')
                                    except BaseException:
                                        time.sleep(pause)
                                    else:
                                        if future_min is None:
                                            continue
                                        else:
                                            break
                                future_min = pd.DataFrame(future_min)
                                if future_min.empty:
                                    log.logger.info("期货合约" + future_code +
                                                    "在" + Start_Date +
                                                    "已过期,请检查!")
                                else:  # 首次出现的合约需要有header
                                    future_min = future_min.sort_values(
                                        by='trade_time')
                                    future_min = future_min.fillna(
                                        method='ffill').fillna(0)
                                    future_min.index = range(
                                        future_min.shape[0])
                                    if future_min['trade_time'].iloc[-1][
                                            -8:] == '00:00:00':
                                        future_min = future_min.drop(
                                            (future_min.shape[0] - 1))
                                    else:
                                        pass
                                    if future_code[-3:] == "CFX":
                                        future_min.to_csv(
                                            DataPath + "Minute\\CFFEX\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            # header=False,
                                            mode='a')
                                    elif future_code[-3:] == "SHF":
                                        future_min.to_csv(
                                            DataPath + "Minute\\SHFE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            # header=False,
                                            mode='a')
                                    elif future_code[-3:] == "DCE":
                                        future_min.to_csv(
                                            DataPath + "Minute\\DCE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            # header=False,
                                            mode='a')
                                    elif future_code[-3:] == "ZCE":
                                        future_min.to_csv(
                                            DataPath + "Minute\\CZCE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            # header=False,
                                            mode='a')
                                    elif future_code[-3:] == "INE":
                                        future_min.to_csv(
                                            DataPath + "Minute\\INE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            # header=False,
                                            mode='a')
                            else:
                                for _ in range(retry_count):
                                    # 加入重试机制，防止访问过于频繁
                                    try:
                                        future_min = ts.pro_bar(
                                            api=pro,
                                            ts_code=future_code,
                                            asset='FT',
                                            adj=None,
                                            start_date=Start_Date,
                                            end_date=End_Date,
                                            freq='1min')
                                    except BaseException:
                                        time.sleep(pause)
                                    else:
                                        if future_min is None:
                                            continue
                                        else:
                                            break
                                future_min = pd.DataFrame(future_min)
                                if future_min.empty:  # 合约是否过期
                                    log.logger.info("期货合约" + future_code +
                                                    "在" + Start_Date +
                                                    "已过期,请检查!")
                                else:  # 首次出现的合约需要有header
                                    future_min = future_min.sort_values(
                                        by='trade_time')
                                    future_min = future_min.fillna(
                                        method='ffill').fillna(0)
                                    future_min.index = range(
                                        future_min.shape[0])
                                    if future_min['trade_time'].iloc[-1][
                                            -8:] == '00:00:00':
                                        future_min = future_min.drop(
                                            (future_min.shape[0] - 1))
                                    else:
                                        pass
                                    if future_code[-3:] == "CFX":
                                        future_min.to_csv(
                                            DataPath + "Minute\\CFFEX\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                                    elif future_code[-3:] == "SHF":
                                        future_min.to_csv(
                                            DataPath + "Minute\\SHFE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                                    elif future_code[-3:] == "DCE":
                                        future_min.to_csv(
                                            DataPath + "Minute\\DCE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                                    elif future_code[-3:] == "ZCE":
                                        future_min.to_csv(
                                            DataPath + "Minute\\CZCE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                                    elif future_code[-3:] == "INE":
                                        future_min.to_csv(
                                            DataPath + "Minute\\INE\\" +
                                            future_code[:-4] + ".csv",
                                            index=False,
                                            header=False,
                                            mode='a')
                    except Exception:
                        log.logger.error("Exception Logged " + future_code)
                else:
                    pass
        return


def minute_func(params):
    pro = ts.pro_api()
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 设置日志文件
    log = Logger(Pre_path + "\\Logging\\T_DataUpdate_Min\\" + 'Min_Error.log',
                 level='info')
    start_date = params['start_date']
    end_date = params['end_date']
    trading_day = params['trading_day']
    data_update = Data_Update(start_date, end_date, pro, log)
    if params['func_name'] == 'stock_min':
        stock_list = params['stock_list']
        data_update.stock_min(trading_day, stock_list)
    else:
        future_list = params['future_list']
        data_update.future_min(trading_day, future_list)
    return


def set_params(code_list, trading_day, start_date, end_date, func_name):
    td = {
        'trading_day': trading_day,
        'start_date': start_date,
        'end_date': end_date,
        'func_name': func_name
    }
    params = []
    if func_name == 'stock_min':
        for i, sec_code in enumerate(code_list):
            td['stock_list'] = sec_code
            params.append(td.copy())
    else:
        for i, sec_code in enumerate(code_list):
            td['future_list'] = sec_code
            params.append(td.copy())
    return params


if __name__ == "__main__":
    """初始化tushare接口"""
    pro = ts.pro_api()
    """建立数据连接"""
    # db = MySQLConn.DBCONN()
    """设置下载日期"""
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    Start_Date = str(TradeDate.iloc[0, 0])
    End_Date = str(TradeDate.iloc[-1, 0])
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 设置日志文件
    log = Logger(Pre_path + "\\Logging\\T_DataUpdate_Min\\" +'Minute_Error.log',level='info')
    # 开始更新数据
    print("Data of minute_date begin to update !")
    data_update = Data_Update(Start_Date, End_Date, pro, log)
    stock_day, stock_list, index_list, future_day, future_list = data_update.basic_data()
    print("The update of basic_data is finished~")
    # 分钟数据需要往后设置一天
    End_Date = str(parse(End_Date) + dt.timedelta(days=1))
    End_Date = End_Date[0:4] + End_Date[5:7] + End_Date[8:10]
    stock_day.append(End_Date)
    future_day.append(End_Date)
    # # ==========================================================
    # # ==========================================================
    stock_list = [stock_list[i:i + 50] for i in range(0, len(stock_list), 50)]
    para_x = set_params(stock_list, stock_day, Start_Date, End_Date,'stock_min')
    pool = multiprocessing.Pool(8)
    # 多进程map中不能有传入地址参数pro、log
    pool.map(minute_func, para_x)
    pool.close()
    pool.join()
    print("The update of stock_min is finished~")
    # # =========================================================
    # # =========================================================
    data_update.index_min(stock_day, index_list)
    print("The update of index_min is finished~")
    # =========================================================
    # =========================================================
    future_list = [future_list[i:i + 50] for i in range(0, len(future_list), 50)]
    para_y = set_params(future_list, future_day, Start_Date, End_Date, 'future_min')
    pool = multiprocessing.Pool(4)
    # 多进程map中不能有传入地址参数pro、log
    pool.map(minute_func, para_y)
    pool.close()
    pool.join()
    print("The update of future_min is finished~")
    # 缺失数据自动填补
    file_list = os.listdir(Pre_path + "MarketData\\Minute")  # 获取股票文件夹列表
    for file in file_list:
        date_list = list(os.listdir(Pre_path + "MarketData\\Minute\\" + file)) # 获取已下载日期列表
        if End_Date not in date_list:
            data_update.stock_min([file + ".SH" if file[0] == "6" else file+".SZ"], End_Date)
        else:
            continue

