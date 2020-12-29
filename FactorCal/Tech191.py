#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/5 15:12
# @Author: Tu Xinglong
# @File  : Tech191.py

import os
import sys
import re
import multiprocessing

import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
import scipy as sp
import tushare as ts
from pyfinance.ols import OLS, PandasRollingOLS

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa


class Tech191(object):
    '''
    国泰君安191个短周期技术因子
    '''

    def __init__(self, startdate, enddate, count, length):
        '''
        获取数据信息
        param df_data:
        '''
        stock_price = get_price(startdate=startdate,
                                enddate=enddate,
                                fields=[
                                    'open_qfq', 'close_qfq', 'low_qfq',
                                    'high_qfq', 'volume', 'amount', 'ret'
                                ],
                                count=count)
        benchmark_price = get_price(startdate=startdate,
                                    enddate=enddate,
                                    fields=['index'],
                                    count=count)
        self.open = stock_price['open_qfq']
        self.close = stock_price['close_qfq']
        self.low = stock_price['low_qfq']
        self.high = stock_price['high_qfq']
        self.volume = stock_price['volume'] * 100
        self.amount = stock_price['amount'] * 1000
        self.returns = stock_price['ret']
        self.benchmark_open = benchmark_price['index']['沪深300open']
        self.benchmark_close = benchmark_price['index']['沪深300close']
        self.vwap = stock_price['amount'] / (stock_price['volume'] + 0.001)
        self.pre_close = self.close.shift(1).fillna(0)
        self.length = length
        np.seterr(divide='ignore', invalid='ignore')  # 忽略警告

        '''
        1.dataframe与dataframe,0,1比较用np.maximum和np.minimum
          与其他数据比较用ts_max和ts_min，属于时间序列
        2.公式中的幂次方计算一律用pow()函数
        3.benchmark_open与benchmark_close为series
        4.逻辑运算符一律用&, |表示
        '''
    # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

    def tech001(self):
        tech001 = (-1*correlation(rank(delta(log(self.volume), 1)),
                                  rank(((self.close-self.open)/self.open)), 6))
        save_hdf(tech001, 'tech001', self.length)
        return

    # (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def tech002(self):
        tech002 = (-1*delta((((self.close-self.low) -
                              (self.high-self.close))/(self.high-self.low)), 1))
        save_hdf(tech002, 'tech002', self.length)
        return

    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    def tech003(self):
        part1 = self.close
        part1[self.close == delay(self.close, 1)] = 0
        part2 = np.maximum(self.high, delay(self.close, 1))
        part2[self.close > delay(self.close, 1)] = np.minimum(
            self.low, delay(self.close, 1))
        tech003 = pd.DataFrame(
            ts_sum(part1-part2, 6), index=self.close.index, columns=self.close.columns)
        save_hdf(tech003, 'tech003', self.length)
        return

    # ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <
    # ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /
    # MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
    def tech004(self):
        cond1 = ((ts_sum(self.close, 8) / 8 + stddev(self.close, 8))
                 < ts_sum(self.close, 2) / 2)
        cond2 = (ts_sum(self.close, 2) / 2 <
                 (ts_sum(self.close, 8) / 8 - stddev(self.close, 8)))
        cond3 = (1 <= self.volume / mean(self.volume, 20))
        tech004 = -1 * pd.DataFrame(np.ones(self.close.shape),
                                    index=self.close.index, columns=self.close.columns)
        tech004[cond1] = -1
        tech004[cond2] = 1
        tech004[cond3] = 1
        save_hdf(tech004, 'tech004', self.length)
        return

    # (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    def tech005(self):
        tech005 = (-1*ts_max(correlation(ts_rank(self.volume, 5),
                                         ts_rank(self.high, 5), 5), 5))
        save_hdf(tech005, 'tech005', self.length)
        return

    # (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    def tech006(self):
        tech006 = (rank(sign(delta((self.open*0.85+self.high*0.15), 4)))*(-1))
        save_hdf(tech006, 'tech006', self.length)
        return

    # ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def tech007(self):
        tech007 = ((rank(ts_max((self.vwap-self.close), 3)) +
                    rank(ts_min((self.vwap-self.close), 3))) *
                   rank(delta(self.volume, 3)))
        save_hdf(tech007, 'tech007', self.length)
        return

    # RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    def tech008(self):
        tech008 = rank(
            delta(((((self.high+self.low)/2)*0.2)+(self.vwap*0.8)), 4)*(-1))
        save_hdf(tech008, 'tech008', self.length)
        return

    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    def tech009(self):
        tech009 = sma(((self.high+self.low)/2-(delay(self.high, 1) +
                                               delay(self.low, 1))/2)*(self.high-self.low)/self.volume, 7, 2)
        save_hdf(tech009, 'tech009', self.length)
        return

    # (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    def tech010(self):
        part1 = self.returns
        part1[self.returns < 0] = stddev(self.returns, 20)
        tech010 = rank(ts_max(pow(part1, 2), 5))
        save_hdf(tech010, 'tech010', self.length)
        return

    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    def tech011(self):
        tech011 = ts_sum(((self.close-self.low)-(self.high -
                                                 self.close))/(self.high-self.low)*self.volume, 6)
        save_hdf(tech011, 'tech011', self.length)
        return

    # (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    def tech012(self):
        tech012 = (rank((self.open - (ts_sum(self.vwap, 10) / 10)))
                   ) * (-1 * (rank(abs((self.close - self.vwap)))))
        save_hdf(tech012, 'tech012', self.length)
        return

    # (((HIGH * LOW)^0.5) - VWAP)
    def tech013(self):
        tech013 = (pow((self.high * self.low), 0.5) - self.vwap)
        save_hdf(tech013, 'tech013', self.length)
        return

    # CLOSE-DELAY(CLOSE,5)
    def tech014(self):
        tech014 = self.close-delay(self.close, 5)
        save_hdf(tech014, 'tech014', self.length)
        return

    # OPEN/DELAY(CLOSE,1)-1
    def tech015(self):
        tech015 = self.open/delay(self.close, 1)-1
        save_hdf(tech015, 'tech015', self.length)
        return

    # (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    def tech016(self):
        tech016 = (-1*ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))
        save_hdf(tech016, 'tech016', self.length)
        return

    # RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
    def tech017(self):
        tech017 = pow(rank((self.vwap - ts_max(self.vwap, 15))),
                      delta(self.close, 5))
        save_hdf(tech017, 'tech017', self.length)
        return

    # CLOSE/DELAY(CLOSE,5)
    def tech018(self):
        tech018 = self.close/delay(self.close, 5)
        save_hdf(tech018, 'tech018', self.length)
        return

    # (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
    def tech019(self):
        cond1 = (self.close < delay(self.close, 5))
        cond2 = (self.close == delay(self.close, 5))
        tech019 = (self.close-delay(self.close, 5))/self.close
        tech019[cond1] = (self.close-delay(self.close, 5))/delay(self.close, 5)
        tech019[cond2] = 0
        save_hdf(tech019, 'tech019', self.length)
        return

    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    def tech020(self):
        tech020 = (self.close-delay(self.close, 6))/delay(self.close, 6)*100
        save_hdf(tech020, 'tech020', self.length)
        return

    # REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    def tech021(self):
        tech021 = regbeta(mean(self.close, 6), sequence(6), 6)
        save_hdf(tech021, 'tech021', self.length)
        return

    # SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def tech022(self):
        tech022 = sma(((self.close-mean(self.close, 6))/mean(self.close, 6) -
                       delay((self.close-mean(self.close, 6))/mean(self.close, 6), 3)), 12, 1)
        save_hdf(tech022, 'tech022', self.length)
        return

    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1 )+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    def tech023(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close <= delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = stddev(self.close, 20)
        part2 = sma(part1, 20, 1)
        part3 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part3[cond2] = stddev(self.close, 20)
        part4 = sma(part3, 20, 1)
        tech023 = part2/(part2+part4)
        save_hdf(tech023, 'tech023', self.length)
        return

    # SMA(CLOSE-DELAY(CLOSE,5),5,1)
    def tech024(self):
        tech024 = sma(self.close-delay(self.close, 5), 5, 1)
        save_hdf(tech024, 'tech024', self.length)
        return

    # ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))
    def tech025(self):
        tech025 = ((-1 * rank((delta(self.close, 7) * (1 - rank(decay_linear((self.volume /
                                                                              mean(self.volume, 20)), 9)))))) * (1 + rank(ts_sum(self.returns, 250))))
        save_hdf(tech025, 'tech025', self.length)
        return

    # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
    def tech026(self):
        tech026 = ((((ts_sum(self.close, 7) / 7) - self.close)) +
                   ((correlation(self.vwap, delay(self.close, 5), 230))))
        save_hdf(tech026, 'tech026', self.length)
        return

    # WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    def tech027(self):
        tech027 = wma((self.close-delay(self.close, 3))/delay(self.close, 3)
                      * 100+(self.close-delay(self.close, 6))/delay(self.close, 6)*100, 12)
        save_hdf(tech027, 'tech027', self.length)
        return

    # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/( MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    def tech028(self):
        tech028 = 3*sma((self.close-ts_min(self.low, 9))/(ts_max(self.high, 9)-ts_min(self.low, 9))*100, 3, 1) - \
            2*sma(sma((self.close-ts_min(self.low, 9)) /
                      (ts_max(self.high, 9)-ts_max(self.low, 9))*100, 3, 1), 3, 1)
        save_hdf(tech028, 'tech028', self.length)
        return

    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def tech029(self):
        tech029 = (self.close-delay(self.close, 6)) / \
            delay(self.close, 6)*self.volume
        save_hdf(tech029, 'tech029', self.length)
        return

    # # WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML，60))^2,20)
    # def tech030(self):
    #     return 0

    # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    def tech031(self):
        tech031 = (self.close-mean(self.close, 12))/mean(self.close, 12)*100
        save_hdf(tech031, 'tech031', self.length)
        return

    # (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    def tech032(self):
        tech032 = (-1*ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3))
        save_hdf(tech032, 'tech032', self.length)
        return

    # ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))
    def tech033(self):
        tech033 = ((((-1*ts_min(self.low, 5))+delay(ts_min(self.low, 5), 5))*rank(
            ((ts_sum(self.returns, 240)-ts_sum(self.returns, 20))/220)))*ts_rank(self.volume, 5))
        save_hdf(tech033, 'tech033', self.length)
        return

    # MEAN(CLOSE,12)/CLOSE
    def tech034(self):
        tech034 = mean(self.close, 12)/self.close
        save_hdf(tech034, 'tech034', self.length)
        return

    # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)
    def tech035(self):
        part1 = (np.minimum(rank(decay_linear(delta(self.open, 1), 15)), rank(decay_linear(
            correlation((self.volume), ((self.open * 0.65) + (self.open * 0.35)), 17), 7))) * -1)
        tech035 = pd.DataFrame(
            part1, index=self.close.index, columns=self.close.columns)
        save_hdf(tech035, 'tech035', self.length)
        return

    # RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP))6, 2))
    def tech036(self):
        tech036 = rank(
            ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2))
        save_hdf(tech036, 'tech036', self.length)
        return

    # (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    def tech037(self):
        tech037 = (-1*rank(((ts_sum(self.open, 5)*ts_sum(self.returns, 5)) -
                            delay((ts_sum(self.open, 5)*ts_sum(self.returns, 5)), 10))))
        save_hdf(tech037, 'tech037', self.length)
        return

    # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    def tech038(self):
        cond1 = ((ts_sum(self.high, 20)/20) < self.high)
        tech038 = pd.DataFrame(np.zeros(self.close.shape),
                               index=self.close.index, columns=self.close.columns)
        tech038[cond1] = (-1*delta(self.high, 2))
        save_hdf(tech038, 'tech038', self.length)
        return

    # ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
    def tech039(self):
        tech039 = ((rank(decay_linear(delta((self.close), 2), 8)) - rank(decay_linear(correlation(
            ((self.vwap * 0.3) + (self.open * 0.7)), ts_sum(mean(self.vwap, 180), 37), 14), 12))) * -1)
        save_hdf(tech039, 'tech039', self.length)
        return

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
    def tech040(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close <= delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = self.volume
        tech040 = ts_sum(part1, 26)/ts_sum(part2, 26)*100
        save_hdf(tech040, 'tech040', self.length)
        return

    # (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
    def tech041(self):
        tech041 = (rank(ts_max(delta((self.vwap), 3), 5)) * -1)
        save_hdf(tech041, 'tech041', self.length)
        return

    # ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
    def tech042(self):
        tech042 = ((-1 * rank(stddev(self.high, 10))) *
                   correlation(self.high, self.volume, 10))
        save_hdf(tech042, 'tech042', self.length)
        return

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
    def tech043(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close < delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part1[cond2] = -self.volume
        tech043 = ts_sum(part1, 6)
        save_hdf(tech043, 'tech043', self.length)
        return

    # (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
    def tech044(self):
        tech044 = (ts_rank(decay_linear(correlation(((self.low)), mean(
            self.volume, 10), 7), 6), 4) + ts_rank(decay_linear(delta((self.vwap), 3), 10), 15))
        save_hdf(tech044, 'tech044', self.length)
        return

    # (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
    def tech045(self):
        tech045 = (rank(delta((((self.close * 0.6) + (self.open * 0.4))), 1))
                   * rank(correlation(self.vwap, mean(self.volume, 150), 15)))
        save_hdf(tech045, 'tech045', self.length)
        return

    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    def tech046(self):
        tech046 = (mean(self.close, 3)+mean(self.close, 6) +
                   mean(self.close, 12)+mean(self.close, 24))/(4*self.close)
        save_hdf(tech046, 'tech046', self.length)
        return

    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    def tech047(self):
        tech047 = sma((ts_max(self.high, 6) - self.close) /
                      (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9, 1)
        save_hdf(tech047, 'tech047', self.length)
        return

    # (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2))))
    # + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
    def tech048(self):
        tech048 = (-1*((ts_rank(((sign((self.close - delay(self.close, 1))) + sign((delay(self.close, 1)
                                                                                    - delay(self.close, 2)))) + sign((delay(self.close, 2) - delay(self.close, 3)))))) *
                       ts_sum(self.volume, 5)) / ts_sum(self.volume, 20))
        save_hdf(tech048, 'tech048', self.length)
        return

    # SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),
    # ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))
    # ?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)
    # <=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    def tech049(self):
        cond1 = (self.high + self.low) <= (delay(self.high, 1) +
                                           delay(self.low, 1))
        cond2 = (self.high + self.low) >= (delay(self.high, 1) +
                                           delay(self.low, 1))
        part1 = np.maximum(abs(self.high - delay(self.high, 1)),
                           abs(self.low - delay(self.low, 1)))
        part1[cond1] = 0
        part2 = np.maximum(abs(self.high - delay(self.high, 1)),
                           abs(self.low - delay(self.low, 1)))
        part2[cond2] = 0
        tech049 = ts_sum(part2, 12) / (ts_sum(part2, 12) + ts_sum(part1, 12))
        save_hdf(tech049, 'tech049', self.length)
        return

    # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),
    # ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)
    # )?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)
    # >=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1))))
    # ,12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HI GH-DELAY(HIGH,1)),
    # ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:
    # MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=
    # (DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    def tech050(self):
        cond1 = (self.high + self.low) <= (delay(self.high, 1) +
                                           delay(self.low, 1))
        cond2 = (self.high + self.low) >= (delay(self.high, 1) +
                                           delay(self.low, 1))
        part1 = np.maximum(abs(self.high - delay(self.high, 1)),
                           abs(self.low - delay(self.low, 1)))
        part1[cond1] = 0
        part2 = np.maximum(abs(self.high - delay(self.high, 1)),
                           abs(self.low - delay(self.low, 1)))
        part2[cond2] = 0
        tech050 = (ts_sum(part1, 12) - ts_sum(part2, 12)) / \
            (ts_sum(part1, 12) + ts_sum(part2, 12))
        save_hdf(tech050, 'tech050', self.length)
        return

    # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),
    # ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))
    # ?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)
    # >=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    def tech051(self):
        cond1 = (self.high+self.low) <= (delay(self.high, 1)+delay(self.low, 1))
        cond2 = (self.high+self.low) >= (delay(self.high, 1)+delay(self.low, 1))
        part1 = np.maximum(abs(self.high-delay(self.high, 1)),
                           abs(self.low-delay(self.low, 1)))
        part1[cond1] = 0
        part2 = np.maximum(abs(self.high-delay(self.high, 1)),
                           abs(self.low-delay(self.low, 1)))
        part2[cond2] = 0
        tech051 = ts_sum(part1, 12)/(ts_sum(part1, 12)+ts_sum(part2, 12))
        save_hdf(tech051, 'tech051', self.length)
        return

    # SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3-LOW,1)),26)* 100
    def tech052(self):
        tech052 = ts_sum(np.maximum(0, self.high-delay((self.high+self.low+self.close)/3, 1)), 26) / \
            ts_sum(np.maximum(
                0, delay((self.high+self.low+self.close)/3-self.low, 1)), 26)*100
        save_hdf(tech052, 'tech052', self.length)
        return

    # COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    def tech053(self):
        cond1 = (self.close > delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        tech053 = count(part1, 12) / 12 * 100
        save_hdf(tech053, 'tech053', self.length)
        return

    # (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    def tech054(self):
        tech054 = (-1*rank((stddev(abs(self.close-self.open)) +
                            (self.close-self.open))+correlation(self.close, self.open, 10)))
        save_hdf(tech054, 'tech054', self.length)
        return

    # SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/
    # ((ABS(HIGH-DELAY(CL OSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))
    # >ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS E,1))/2
    # +ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))
    # & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+
    # ABS(HIGH-DELAY(CLO SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))
    # +ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    def tech055(self):
        cond1 = (abs(self.high - delay(self.close, 1)) > abs(self.low - delay(self.close, 1))
                 ) & (abs(self.high - delay(self.close, 1)) > abs(self.high - delay(self.low, 1)))
        cond2 = (abs(self.low - delay(self.close, 1)) > abs(self.high - delay(self.low, 1))
                 ) & (abs(self.low - delay(self.close, 1)) > abs(self.high - delay(self.close, 1)))
        part1 = abs(self.high - delay(self.low, 1)) + \
            abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        part1[cond1] = abs(self.high - delay(self.close, 1)) + abs(self.low - delay(
            self.close, 1)) / 2 + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        part1[cond2] = abs(self.low - delay(self.close, 1)) + abs(self.high - delay(
            self.close, 1)) / 2 + abs(delay(self.close, 1) - delay(self.open, 1)) / 4
        tech055 = ts_sum(16 * (self.close - delay(self.close, 1) + (self.close - self.open) / 2 + delay(self.close, 1) - delay(
            self.open, 1)) / part1 * np.maximum(abs(self.high - delay(self.close, 1)), abs(self.low - delay(self.close, 1))), 20)
        save_hdf(tech055, 'tech055', self.length)
        return

    # (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))
    def tech056(self):
        tech056 = ((rank((self.open - ts_min(self.open, 12))) <
                    rank(pow(rank(correlation(ts_sum(((self.high + self.low) / 2), 19), ts_sum(mean(self.volume, 40), 19), 13)), 5)))*1)
        save_hdf(tech056, 'tech056', self.length)
        return

    # SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    def tech057(self):
        tech057 = sma((self.close-ts_min(self.low, 9)) /
                      (ts_max(self.high, 9))*100, 3, 1)
        save_hdf(tech057, 'tech057', self.length)
        return

    # COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    def tech058(self):
        cond1 = (self.close > delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        tech058 = count(part1, 20)/20*100
        save_hdf(tech058, 'tech058', self.length)
        return

    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,D ELAY(CLOSE,1)))),20)
    def tech059(self):
        cond1 = (self.close == delay(self.close, 1))
        cond2 = (self.close > delay(self.close, 1))
        part1 = self.close
        part1[cond1] = 0
        part2 = np.maximum(self.high, delay(self.close, 1))
        part2[cond2] = np.minimum(self.low, delay(self.close, 1))
        tech059 = ts_sum(part1-part2, 20)
        save_hdf(tech059, 'tech059', self.length)
        return

    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
    def tech060(self):
        tech060 = ts_sum(((self.close-self.low)-(self.high -
                                                 self.close))/(self.high-self.low)*self.volume, 20)
        save_hdf(tech060, 'tech060', self.length)
        return

    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,80), 8)), 17))) * -1)
    def tech061(self):
        tech061 = (np.maximum(rank(decay_linear(delta(self.vwap, 1), 12)), rank(
            decay_linear(rank(correlation(self.low, mean(self.volume, 80), 8)), 17))) * -1)
        save_hdf(tech061, 'tech061', self.length)
        return

    # (-1 * CORR(HIGH, RANK(VOLUME), 5))
    def tech062(self):
        tech062 = (-1*correlation(self.high, rank(self.volume), 5))
        save_hdf(tech062, 'tech062', self.length)
        return

    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    def tech063(self):
        tech063 = sma(np.maximum(self.close-delay(self.close, 1), 0),
                      6, 1)/sma(abs(self.close-delay(self.close, 1)), 6, 1)*100
        save_hdf(tech063, 'tech063', self.length)
        return

    # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,64)), 4), 13), 14))) * -1)
    def tech064(self):
        tech064 = (np.maximum(rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4)), rank(
            decay_linear(ts_max(correlation(rank(self.close), rank(mean(self.volume, 64)), 4), 13), 14))) * -1)
        save_hdf(tech064, 'tech064', self.length)
        return

    # MEAN(CLOSE,6)/CLOSE
    def tech065(self):
        tech065 = mean(self.close, 6)/self.close
        save_hdf(tech065, 'tech065', self.length)
        return

    # (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    def tech066(self):
        tech066 = (self.close-mean(self.close, 6))/mean(self.close, 6)*100
        save_hdf(tech066, 'tech066', self.length)
        return

    ##################################################################
    def tech067(self):
        tech067 = sma(np.maximum(self.close-delay(self.close, 1), 0),
                      24, 1)/sma(abs(self.close-delay(self.close, 1)), 24, 1)*100
        save_hdf(tech067, 'tech067', self.length)
        return

    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    def tech068(self):
        tech068 = sma(((self.high+self.low)/2-(delay(self.high, 1) +
                                               delay(self.low, 1))/2)*(self.high-self.low)/self.volume, 15, 2)
        save_hdf(tech068, 'tech068', self.length)
        return

    # (SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)？0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    def tech069(self):
        cond1 = (self.open <= delay(self.open, 1))
        DTM = np.maximum((self.high-self.open),
                         (self.open-delay(self.open, 1)))
        DTM[cond1] = 0
        cond2 = (self.open >= delay(self.open, 1))
        DBM = np.maximum((self.open-self.low), (self.open-delay(self.open, 1)))
        DBM[cond2] = 0
        cond3 = ts_sum(DTM, 20) > ts_sum(DBM, 20)
        cond4 = ts_sum(DTM, 20) == ts_sum(DBM, 20)
        tech069 = (ts_sum(DTM, 20)-ts_sum(DBM, 20))/ts_sum(DBM, 20)
        tech069[cond3] = (ts_sum(DTM, 20)-ts_sum(DBM, 20))/ts_sum(DTM, 20)
        tech069[cond4] = 0
        save_hdf(tech069, 'tech069', self.length)
        return

    # STD(AMOUNT, 6)
    def tech070(self):
        tech070 = stddev(self.amount, 6)
        save_hdf(tech070, 'tech070', self.length)
        return

    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    def tech071(self):
        tech071 = (self.close-mean(self.close, 24))/mean(self.close, 24)*100
        save_hdf(tech071, 'tech071', self.length)
        return

    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    def tech072(self):
        tech072 = sma((ts_max(self.high, 6)-self.close) /
                      (ts_max(self.high, 6)-ts_min(self.low, 6))*100, 15, 1)
        save_hdf(tech072, 'tech072', self.length)
        return

    # ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
    def tech073(self):
        tech073 = ((ts_rank(decay_linear(decay_linear(correlation((self.close), self.volume, 10), 16), 4),
                            5) - rank(decay_linear(correlation(self.vwap, mean(self.volume, 30), 4), 3))) * -1)
        save_hdf(tech073, 'tech073', self.length)
        return

    # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
    def tech074(self):
        tech074 = (rank(correlation(ts_sum(((self.low * 0.35) + (self.vwap * 0.65)), 20), ts_sum(
            mean(self.volume, 40), 20), 7)) + rank(correlation(rank(self.vwap), rank(self.volume), 6)))
        save_hdf(tech074, 'tech074', self.length)
        return

    # COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
    def tech075(self):
        cond1 = (self.close > self.open) & (pd.DataFrame(np.tile((self.benchmark_close < self.benchmark_open),
                                                                 (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        cond2 = (self.benchmark_close < self.benchmark_open)
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = 1
        tech075 = count(part1, 50)/count(part2, 50)
        save_hdf(tech075, 'tech075', self.length)
        return

    # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    def tech076(self):
        tech076 = stddev(abs((self.close/delay(self.close, 1)-1))/self.volume, 20) / \
            mean(abs((self.close/delay(self.close, 1)-1))/self.volume, 20)
        save_hdf(tech076, 'tech076', self.length)
        return

    # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6))
    def tech077(self):
        tech077 = np.minimum(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)),
                             rank(decay_linear(correlation(((self.high + self.low) / 2), mean(self.volume, 40), 3), 6)))
        save_hdf(tech077, 'tech077', self.length)
        return

    # ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    def tech078(self):
        tech078 = ((self.high+self.low+self.close)/3-mean((self.high+self.low+self.close)/3, 12)) / \
            (0.015*mean(abs(self.close-mean((self.high+self.low+self.close)/3, 12)), 12))
        save_hdf(tech078, 'tech078', self.length)
        return

    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    def tech079(self):
        tech079 = sma(np.maximum(self.close-delay(self.close, 1), 0),
                      12, 1)/sma(abs(self.close-delay(self.close, 1)), 12, 1)*100
        save_hdf(tech079, 'tech079', self.length)
        return

    # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    def tech080(self):
        tech080 = (self.volume-delay(self.volume, 5))/delay(self.volume, 5)*100
        save_hdf(tech080, 'tech080', self.length)
        return

    # SMA(VOLUME,21,2)
    def tech081(self):
        tech081 = sma(self.volume, 21, 2)
        save_hdf(tech081, 'tech081', self.length)
        return

    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    def tech082(self):
        tech082 = sma((ts_max(self.high, 6)-self.close) /
                      (ts_max(self.high, 6)-ts_min(self.low, 6))*100, 20, 1)
        save_hdf(tech082, 'tech082', self.length)
        return

    # (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    def tech083(self):
        tech083 = (-1 * rank(covariance(rank(self.high), rank(self.volume), 5)))
        save_hdf(tech083, 'tech083', self.length)
        return

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    def tech084(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close < delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part1[cond2] = -self.volume
        tech084 = ts_sum(part1, 20)
        save_hdf(tech084, 'tech084', self.length)
        return

    # (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
    def tech085(self):
        tech085 = (ts_rank((self.volume / mean(self.volume, 20)), 20)
                   * ts_rank((-1 * delta(self.close, 7)), 8))
        save_hdf(tech085, 'tech085', self.length)
        return

    # ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)))
    # ? (-1 * 1) :(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) /
    # 10)) < 0) ? 1 :((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))
    def tech086(self):
        cond1 = (0.25 < (((delay(self.close, 20) - delay(self.close, 10)
                           ) / 10) - ((delay(self.close, 10) - self.close) / 10)))
        cond2 = ((((delay(self.close, 20) - delay(self.close, 10)) /
                   10) - ((delay(self.close, 10) - self.close) / 10)) < 0)
        tech086 = ((-1 * 1) * (self.close - delay(self.close, 1)))
        tech086[cond1] = -1 * 1
        tech086[cond2] = 1
        save_hdf(tech086, 'tech086', self.length)
        return

    # ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
    def tech087(self):
        tech087 = ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(((((self.low * 0.9) +
                                                                                         (self.low * 0.1)) - self.vwap) / (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
        save_hdf(tech087, 'tech087', self.length)
        return

    # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100

    def tech088(self):
        tech088 = (self.close-delay(self.close, 20))/delay(self.close, 20)*100
        save_hdf(tech088, 'tech088', self.length)
        return

    # 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    def tech089(self):
        tech089 = 2*(sma(self.close, 13, 2)-sma(self.close, 27, 2) -
                     sma(sma(self.close, 13, 2)-sma(self.close, 27, 2), 10, 2))
        save_hdf(tech089, 'tech089', self.length)
        return

    # ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
    def tech090(self):
        tech090 = (rank(correlation(rank(self.vwap), rank(self.volume), 5)) * -1)
        save_hdf(tech090, 'tech090', self.length)
        return

    # ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
    def tech091(self):
        tech091 = ((rank((self.close - ts_max(self.close, 5))) *
                    rank(correlation((mean(self.volume, 40)), self.low, 5))) * -1)
        save_hdf(tech091, 'tech091', self.length)
        return

    # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
    def tech092(self):
        tech092 = (np.maximum(rank(decay_linear(delta(((self.close*0.35)+(self.vwap*0.65)), 2), 3)),
                              ts_rank(decay_linear(abs(correlation((mean(self.volume, 180)), self.close, 13)), 5), 15))*-1)
        save_hdf(tech092, 'tech092', self.length)
        return

    # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    def tech093(self):
        cond1 = (self.open >= delay(self.open, 1))
        part1 = np.maximum((self.open-self.low),
                           (self.open-delay(self.open, 1)))
        part1[cond1] = 0
        tech093 = ts_sum(part1, 20)
        save_hdf(tech093, 'tech093', self.length)
        return

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    def tech094(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close < delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part1[cond2] = -self.volume
        tech094 = ts_sum(part1, 30)
        save_hdf(tech094, 'tech094', self.length)
        return

    # STD(AMOUNT,20)
    def tech095(self):
        tech095 = stddev(self.amount, 20)
        save_hdf(tech095, 'tech095', self.length)
        return
    # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)

    def tech096(self):
        tech096 = sma(sma((self.close-ts_min(self.low, 9)) /
                          (ts_max(self.high, 9)-ts_min(self.low, 9))*100, 3, 1), 3, 1)
        save_hdf(tech096, 'tech096', self.length)
        return

    # STD(VOLUME,10)
    def tech097(self):
        tech097 = stddev(self.volume, 10)
        save_hdf(tech097, 'tech097', self.length)
        return

    # ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
    def tech098(self):
        cond1 = ((delta((ts_sum(self.close, 100)/100), 100) /
                  delta(self.close, 100)) <= 0.05)
        tech098 = (-1*delta(self.close, 3))
        tech098[cond1] = (-1 * (self.close - ts_min(self.close, 100)))
        save_hdf(tech098, 'tech098', self.length)
        return

    # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    def tech099(self):
        tech099 = (-1 * rank(covariance(rank(self.close), rank(self.volume), 5)))
        save_hdf(tech099, 'tech099', self.length)
        return

    # STD(VOLUME,20)
    def tech100(self):
        tech100 = stddev(self.volume, 20)
        save_hdf(tech100, 'tech100', self.length)
        return

    # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1)
    def tech101(self):
        tech101 = ((rank(correlation(self.close, ts_sum(mean(self.volume, 30), 37), 15)) < rank(
            correlation(rank(((self.high*0.1)+(self.vwap*0.9))), rank(self.volume), 11)))*-1)
        save_hdf(tech101, 'tech101', self.length)
        return

    # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    def tech102(self):
        tech102 = sma(np.maximum(self.volume-delay(self.volume, 1), 0),
                      6, 1)/sma(abs(self.volume-delay(self.volume, 1)), 6, 1)*100
        save_hdf(tech102, 'tech102', self.length)
        return

    # ((20-LOWDAY(LOW,20))/20)*100
    def tech103(self):
        tech103 = ((20-lowday(self.low, 20))/20)*100
        save_hdf(tech103, 'tech103', self.length)
        return

    # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
    def tech104(self):
        tech104 = (-1*(delta(correlation(self.high, self.volume, 5), 5)
                       * rank(stddev(self.close, 20))))
        save_hdf(tech104, 'tech104', self.length)
        return

    # (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
    def tech105(self):
        tech105 = (-1*correlation(rank(self.open), rank(self.volume), 10))
        save_hdf(tech105, 'tech105', self.length)
        return

    # CLOSE-DELAY(CLOSE,20)
    def tech106(self):
        tech106 = self.close-delay(self.close, 20)
        save_hdf(tech106, 'tech106', self.length)
        return

    # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
    def tech107(self):
        tech107 = (((-1*rank((self.open-delay(self.high, 1))))*rank((self.open -
                                                                     delay(self.close, 1))))*rank((self.open-delay(self.low, 1))))
        save_hdf(tech107, 'tech107', self.length)
        return

    # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1)
    def tech108(self):
        tech108 = (pow(rank((self.high-ts_min(self.high, 2))),
                       rank(correlation((self.vwap), (mean(self.volume, 120)), 6)))*-1)
        save_hdf(tech108, 'tech108', self.length)
        return

    # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
    def tech109(self):
        tech109 = sma(self.high-self.low, 10, 2) / \
            sma(sma(self.high-self.low, 10, 2), 10, 2)
        save_hdf(tech109, 'tech109', self.length)
        return

    # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    def tech110(self):
        tech110 = ts_sum(np.maximum(0, self.high-delay(self.close, 1)), 20) / \
            ts_sum(np.maximum(0, delay(self.close, 1)-self.low), 20)*100
        save_hdf(tech110, 'tech110', self.length)
        return

    # SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-L OW),4,2)
    def tech111(self):
        tech111 = sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low), 11, 2) - \
            sma(self.volume*((self.close-self.low) -
                             (self.high-self.close))/(self.high-self.low), 4, 2)
        save_hdf(tech111, 'tech111', self.length)
        return

    # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    def tech112(self):
        cond1 = ((self.close - delay(self.close, 1)) > 0)
        cond2 = ((self.close - delay(self.close, 1)) < 0)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = (self.close - delay(self.close, 1))
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = abs(self.close - delay(self.close, 1))
        tech112 = (ts_sum(part1, 12)-ts_sum(part2, 12)) / \
            (ts_sum(part1, 12)+ts_sum(part2, 12)) * 100
        save_hdf(tech112, 'tech112', self.length)
        return

    # (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))
    def tech113(self):
        tech113 = (-1 * ((rank((ts_sum(delay(self.close, 5), 20) / 20)) * correlation(self.close,
                                                                                      self.volume, 2)) * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))))
        save_hdf(tech113, 'tech113', self.length)
        return

    # ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    def tech114(self):
        tech114 = ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))
                    ) / (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))
        save_hdf(tech114, 'tech114', self.length)
        return

    # RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7))
    def tech115(self):
        tech115 = pow(rank(correlation(((self.high*0.9)+(self.close*0.1)), mean(self.volume, 30), 10)),
                      rank(correlation(ts_rank(((self.high+self.low)/2), 4), ts_rank(self.volume, 10), 7)))
        save_hdf(tech115, 'tech115', self.length)
        return

    # REGBETA(CLOSE,SEQUENCE,20) #
    def tech116(self):
        tech116 = regbeta(self.close, sequence(20), 20)
        save_hdf(tech116, 'tech116', self.length)
        return

   # ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
    def tech117(self):
        tech117 = ((ts_rank(self.volume, 32) * (1 - ts_rank(((self.close +
                                                              self.high) - self.low), 16))) * (1 - ts_rank(self.returns, 32)))
        save_hdf(tech117, 'tech117', self.length)
        return

    # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    def tech118(self):
        tech118 = ts_sum(self.high-self.open, 20) / \
            ts_sum(self.open-self.low, 20)*100
        save_hdf(tech118, 'tech118', self.length)
        return

    # (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) -RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
    def tech119(self):
        tech119 = (rank(decay_linear(correlation(self.vwap, ts_sum(mean(self.volume, 5), 26), 5), 7)) - rank(
            decay_linear(ts_rank(ts_min(correlation(rank(self.open), rank(mean(self.volume, 15)), 21), 9), 7), 8)))
        save_hdf(tech119, 'tech119', self.length)
        return

    # (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    def tech120(self):
        tech120 = (rank((self.vwap - self.close)) /
                   rank((self.vwap + self.close)))
        save_hdf(tech120, 'tech120', self.length)
        return

    # ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)
    def tech121(self):
        tech121 = (pow(rank((self.vwap - ts_min(self.vwap, 12))), ts_rank(correlation(
            ts_rank(self.vwap, 20), ts_rank(mean(self.volume, 60), 2), 18), 3)) * -1)
        save_hdf(tech121, 'tech121', self.length)
        return

    # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SM A(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    def tech122(self):
        tech122 = (sma(sma(sma(log(self.close), 13, 2), 13, 2), 13, 2)-delay(sma(sma(sma(log(self.close),
                                                                                         13, 2), 13, 2), 13, 2), 1))/delay(sma(sma(sma(log(self.close), 13, 2), 13, 2), 13, 2), 1)
        save_hdf(tech122, 'tech122', self.length)
        return

    # ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
    def tech123(self):
        tech123 = ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(
            mean(self.volume, 60), 20), 9)) < rank(correlation(self.low, self.volume, 6))) * -1)
        save_hdf(tech123, 'tech123', self.length)
        return

    # (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
    def tech124(self):
        tech124 = (self.close - self.vwap) / \
            decay_linear(rank(ts_max(self.close, 30)), 2)
        save_hdf(tech124, 'tech124', self.length)
        return

    # (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))
    def tech125(self):
        tech125 = (rank(decay_linear(correlation((self.vwap), mean(self.volume, 80), 17), 20)
                        ) / rank(decay_linear(delta(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)))
        save_hdf(tech125, 'tech125', self.length)
        return

    # (CLOSE + HIGH + LOW) / 3
    def tech126(self):
        tech126 = (self.close+self.high+self.low)/3
        save_hdf(tech126, 'tech126', self.length)
        return

    # (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
    def tech127(self):
        tech127 = pow(mean(pow(
            100*(self.close-ts_max(self.close, 12))/(ts_max(self.close, 12)), 2), 12), 0.5)
        save_hdf(tech127, 'tech127', self.length)
        return

    # 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0), 14)))
    def tech128(self):
        cond1 = (self.high+self.low+self.close) / \
            3 > delay((self.high+self.low+self.close)/3, 1)
        cond2 = (self.high + self.low + self.close) / \
            3 < delay((self.high + self.low + self.close) / 3, 1)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = (self.high+self.low+self.close)/3*self.volume
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = (self.high+self.low+self.close)/3*self.volume
        tech128 = 100-(100/(1+ts_sum(part1, 14)/ts_sum(part2, 14)))
        save_hdf(tech128, 'tech128', self.length)
        return

    # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    def tech129(self):
        cond1 = ((self.close-delay(self.close, 1)) < 0)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = abs(self.close-delay(self.close, 1))
        tech129 = ts_sum(part1, 12)
        save_hdf(tech129, 'tech129', self.length)
        return

    # (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
    def tech130(self):
        tech130 = (rank(decay_linear(correlation(((self.high + self.low) / 2), mean(self.volume, 40), 9),
                                     10)) / rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 7), 3)))
        save_hdf(tech130, 'tech130', self.length)
        return

    # (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
    def tech131(self):
        tech131 = pow(rank(delta(self.vwap, 1)), ts_rank(
            correlation(self.close, mean(self.volume, 50), 18), 18))
        save_hdf(tech131, 'tech131', self.length)
        return

    # MEAN(AMOUNT, 20)
    def tech132(self):
        tech132 = mean(self.amount, 20)
        save_hdf(tech132, 'tech132', self.length)
        return

    # ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    def tech133(self):
        tech133 = ((20-highday(self.high, 20))/20) * \
            100-((20-lowday(self.low, 20))/20)*100
        save_hdf(tech133, 'tech133', self.length)
        return

    # (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    def tech134(self):
        tech134 = (self.close-delay(self.close, 12)) / \
            delay(self.close, 12)*self.volume
        save_hdf(tech134, 'tech134', self.length)
        return

    # SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    def tech135(self):
        tech135 = sma(delay(self.close/delay(self.close, 20), 1), 20, 1)
        save_hdf(tech135, 'tech135', self.length)
        return

    # ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    def tech136(self):
        tech136 = ((-1 * rank(delta(self.returns, 3))) *
                   correlation(self.open, self.volume, 10))
        save_hdf(tech136, 'tech136', self.length)
        return

    # 16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE, 1))>
    # ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))? ABS(HIGH-DELAY(CLOSE,1))+
    # ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>
    # ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+
    # ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+
    # ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    def tech137(self):
        cond1 = (abs(self.high-delay(self.close, 1)) > abs(self.low-delay(self.close, 1))
                 ) & (abs(self.high-delay(self.close, 1)) > abs(self.high-delay(self.low, 1)))
        cond2 = (abs(self.low-delay(self.close, 1)) > abs(self.high-delay(self.low, 1))
                 ) & (abs(self.low-delay(self.close, 1)) > abs(self.high-delay(self.close, 1)))
        part1 = abs(self.high-delay(self.low, 1)) + \
            abs(delay(self.close, 1)-delay(self.open, 1))/4
        part1[cond1] = abs(self.high-delay(self.close, 1))+abs(self.low -
                                                               delay(self.close, 1))/2+abs(delay(self.close, 1)-delay(self.open, 1))/4
        part1[cond2] = abs(self.low-delay(self.close, 1))+abs(self.high -
                                                              delay(self.close, 1))/2+abs(delay(self.close, 1)-delay(self.open, 1))/4
        tech137 = 16*(self.close-delay(self.close, 1)+(self.close-self.open)/2+delay(self.close, 1)-delay(
            self.open, 1))/part1 * np.maximum(abs(self.high-delay(self.close, 1)), abs(self.low-delay(self.close, 1)))
        save_hdf(tech137, 'tech137', self.length)
        return

    # ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
    def tech138(self):
        tech138 = ((rank(decay_linear(delta((((self.low * 0.7) + (self.vwap * 0.3))), 3), 20)) - ts_rank(decay_linear(
            ts_rank(correlation(ts_rank(self.low, 8), ts_rank(mean(self.volume, 60), 17), 5), 19), 16), 7)) * -1)
        save_hdf(tech138, 'tech138', self.length)
        return

    # (-1 * CORR(OPEN, VOLUME, 10))
    def tech139(self):
        tech139 = (-1 * correlation(self.open, self.volume, 10))
        save_hdf(tech139, 'tech139', self.length)
        return

    # MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
    def tech140(self):
        tech140 = np.minimum(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8)),
                             ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(mean(self.volume, 60), 20), 8), 7), 3))
        save_hdf(tech140, 'tech140', self.length)
        return

    # (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
    def tech141(self):
        tech141 = (
            rank(correlation(rank(self.high), rank(mean(self.volume, 15)), 9)) * -1)
        save_hdf(tech141, 'tech141', self.length)
        return

    # (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
    def tech142(self):
        tech142 = (((-1 * rank(ts_rank(self.close, 10))) * rank(delta(delta(self.close, 1), 1)))
                   * rank(ts_rank((self.volume/mean(self.volume, 20)), 5)))
        save_hdf(tech142, 'tech142', self.length)
        return

    # # CLOSE > DELAY(CLOSE, 1)?(CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1) * SELF : SELF
    # def tech143(self):
    #
    #     return 0

    # SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
    def tech144(self):
        cond1 = self.close >= delay(self.close, 1)
        part1 = abs(self.close/delay(self.close, 1) - 1)/self.amount
        part1[cond1] = 0
        cond2 = self.close < delay(self.close, 1)
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = 1
        tech144 = ts_sum(part1, 20)/count(part2, 20)
        save_hdf(tech144, 'tech144', self.length)
        return

    # (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
    def tech145(self):
        tech145 = (mean(self.volume, 9) - mean(self.volume, 26)) / \
            mean(self.volume, 12) * 100
        save_hdf(tech145, 'tech145', self.length)
        return

    # MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*(( CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOS E-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE, 1))/DELAY(CLOSE,1),61,2)))^2,60)
    def tech146(self):
        tech146 = (mean((self.close-delay(self.close, 1))/delay(self.close, 1)-sma((self.close-delay(self.close, 1))/delay(self.close, 1), 61, 2), 20) *
                   ((self.close-delay(self.close, 1))/delay(self.close, 1)-sma((self.close-delay(self.close, 1))/delay(self.close, 1), 61, 2)) /
                   sma(pow(((self.close-delay(self.close, 1))/delay(self.close, 1)-((self.close-delay(self.close, 1))/delay(self.close, 1)-sma((self.close-delay(self.close, 1))/delay(self.close, 1), 61, 2))), 2), 61, 2))
        save_hdf(tech146, 'tech146', self.length)
        return

    # REGBETA(MEAN(CLOSE, 12), SEQUENCE(12))
    def tech147(self):
        tech147 = regbeta(mean(self.close, 12), sequence(12), 12)
        save_hdf(tech147, 'tech147', self.length)
        return

    # ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    def tech148(self):
        tech148 = ((rank(correlation((self.open), ts_sum(mean(self.volume, 60), 9), 6)) < rank(
            (self.open - ts_min(self.open, 14)))) * -1)
        save_hdf(tech148, 'tech148', self.length)
        return

    # #REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1) ),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELA Y(BANCHMARKINDEXCLOSE,1)),252)
    # def tech149(self):
    #     cond1 = self.benchmark_close>=delay(self.benchmark_close,1)
    #     part1 = (self.close/delay(self.close,1)-1).fillna(0).replace([np.inf,-np.inf],[0,0])
    #     part1[cond1] = np.nan
    #     part2 = (self.benchmark_close/delay(self.benchmark_close,1)-1).fillna(0).replace([np.inf,-np.inf],[0,0])
    #     part2[cond1] = np.nan
    #     rows = self.returns.shape[0]
    #     columns = self.returns.columns
    #     tech149 = pd.DataFrame(np.zeros(self.returns.shape) * np.nan,index=self.close.index, columns=self.close.columns)
    #     for i, col in enumerate(columns):
    #         print(i)
    #         for j in range(2500, rows):
    #             print(j)
    #             model = OLS(x=np.array(part2[(j - 500):j].dropna()),
    #                         y=np.array(part1[col][(j - 500):j].dropna()))
    #             tech149.iloc[j, i] = model.beta
    #     save_hdf(tech149, 'tech149')
    #     return

    # (CLOSE + HIGH + LOW) / 3 * VOLUME
    def tech150(self):
        tech150 = (self.close+self.high+self.low)/3*self.volume
        save_hdf(tech150, 'tech150', self.length)
        return

    # SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)
    def tech151(self):
        tech151 = sma(self.close - delay(self.close, 20), 20, 1)
        save_hdf(tech151, 'tech151', self.length)
        return

    # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
    def tech152(self):
        tech152 = sma(mean(delay(sma(delay(self.close/delay(self.close, 9), 1), 9, 1), 1), 12) -
                      mean(delay(sma(delay(self.close/delay(self.close, 9), 1), 9, 1), 1), 26), 9, 1)
        save_hdf(tech152, 'tech152', self.length)
        return

    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    def tech153(self):
        tech153 = (mean(self.close, 3)+mean(self.close, 6) +
                   mean(self.close, 12)+mean(self.close, 24))/4
        save_hdf(tech153, 'tech153', self.length)
        return

    # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))
    def tech154(self):
        tech154 = (((self.vwap-ts_min(self.vwap, 16)) <
                    (correlation(self.vwap, mean(self.vwap, 180), 18)))*1)
        save_hdf(tech154, 'tech154', self.length)
        return

    # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    def tech155(self):
        tech155 = sma(self.volume, 13, 2)-sma(self.volume, 27, 2) - \
            sma(sma(self.volume, 13, 2)-sma(self.volume, 27, 2), 10, 2)
        save_hdf(tech155, 'tech155', self.length)
        return

    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1
    def tech156(self):
        tech156 = (np.maximum(rank(decay_linear(delta(self.vwap, 5), 3)), rank(decay_linear(
            ((delta(((self.open*0.15)+(self.low*0.85)), 2)/((self.open*0.15)+(self.low*0.85)))*-1), 3)))*-1)
        save_hdf(tech156, 'tech156', self.length)
        return

    # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5))
    def tech157(self):
        tech157 = (ts_min(product(rank(rank(log(ts_sum(ts_min(rank(rank(
            (-1*rank(delta((self.close-1), 5))))), 2), 1)))), 1), 5)+ts_rank(delay((-1*self.returns), 6), 5))
        save_hdf(tech157, 'tech157', self.length)
        return

    # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    def tech158(self):
        tech158 = ((self.high-sma(self.close, 15, 2)) -
                   (self.low-sma(self.close, 15, 2)))/self.close
        save_hdf(tech158, 'tech158', self.length)
        return

    # ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6) *12*24+
    # (CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CL OSE,1)),12)*6*24+
    # (CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D ELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    def tech159(self):
        tech159 = (((self.close-ts_sum(np.minimum(self.low, delay(self.close, 1)), 6)) /
                    ts_sum(np.maximum(self.high, delay(self.close, 1))-np.minimum(self.low, delay(self.close, 1)), 6) * 12*24 +
                    (self.close-ts_sum(np.minimum(self.low, delay(self.close, 1)), 12)) /
                    ts_sum(np.maximum(self.high, delay(self.close, 1))-np.minimum(self.low, delay(self.close, 1)), 12)*6*24 +
                    (self.close-ts_sum(np.minimum(self.low, delay(self.close, 1)), 24)) /
                    ts_sum(np.maximum(self.high, delay(self.close, 1))-np.minimum(self.low, delay(self.close, 1)), 24)*6*24)*100 /
                   (6*12+6*24+12*24))
        save_hdf(tech159, 'tech159', self.length)
        return

    # SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def tech160(self):
        cond1 = self.close <= delay(self.close, 1)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = stddev(self.close, 20)
        tech160 = sma(part1, 20, 1)
        save_hdf(tech160, 'tech160', self.length)
        return

    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    def tech161(self):
        tech161 = mean(np.maximum(np.maximum((self.high-self.low), abs(
            delay(self.close, 1)-self.high)), abs(delay(self.close, 1)-self.low)), 12)
        save_hdf(tech161, 'tech161', self.length)
        return

    # (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOS E-DELAY(CLOSE,1),0),12,1)/
    # SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(C LOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-
    # MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    def tech162(self):
        tech162 = ((sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100 -
                    ts_min(sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100, 12)) /
                   (ts_max(sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100, 12) -
                    ts_min(sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100, 12)))
        save_hdf(tech162, 'tech162', self.length)
        return

    # RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    def tech163(self):
        tech163 = rank(((((-1 * self.returns) * mean(self.volume, 20))
                         * self.vwap) * (self.high - self.close)))
        save_hdf(tech163, 'tech163', self.length)
        return

    # SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-D ELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
    def tech164(self):
        cond1 = self.close > delay(self.close, 1)
        part1 = pd.DataFrame(np.ones(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1/(self.close-delay(self.close, 1))
        tech164 = sma((part1-ts_min(part1, 12)) /
                      (self.high-self.low)*100, 13, 2)
        save_hdf(tech164, 'tech164', self.length)
        return

    # # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    # def tech165(self):
    #     # tech165 = np.maximum(np.cumsum(self.close-mean(self.close,48)),0)-np.minimum(np.cumsum(self.close-mean(self.close,48)),0)/stddev(self.close,48)
    #     return 0

    # -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    def tech166(self):
        tech166 = -20*pow(20-1, 1.5)*ts_sum(self.close/delay(self.close, 1)-1-mean(self.close/delay(self.close, 1)-1,
                                                                                   20), 20)/((20-1)*(20-2)*pow(ts_sum(pow(self.close/delay(self.close, 1), 20, 2), 20), 1.5))
        save_hdf(tech166, 'tech166', self.length)
        return

    # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    def tech167(self):
        cond1 = (self.close-delay(self.close, 1)) > 0
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.close-delay(self.close, 1)
        tech167 = ts_sum(part1, 12)
        save_hdf(tech167, 'tech167', self.length)
        return

    # (-1*VOLUME/MEAN(VOLUME,20))
    def tech168(self):
        tech168 = (-1*self.volume/mean(self.volume, 20))
        save_hdf(tech168, 'tech168', self.length)
        return

    # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)
    def tech169(self):
        tech169 = sma(mean(delay(sma(self.close-delay(self.close, 1), 9, 1), 1), 12) -
                      mean(delay(sma(self.close-delay(self.close, 1), 9, 1), 1), 26), 10, 1)
        save_hdf(tech169, 'tech169', self.length)
        return

    # ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))
    def tech170(self):
        tech170 = ((((rank((1 / self.close)) * self.volume) / mean(self.volume, 20)) * ((self.high * rank(
            (self.high - self.close))) / (ts_sum(self.high, 5) / 5))) - rank((self.vwap - delay(self.vwap, 5))))
        save_hdf(tech170, 'tech170', self.length)
        return

    # ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    def tech171(self):
        tech171 = ((-1 * ((self.low - self.close) * pow(self.open, 5))
                    ) / ((self.close - self.high) * pow(self.close, 5)))
        save_hdf(tech171, 'tech171', self.length)
        return

    # MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/(SUM((LD>0&LD>HD)?LD:0,14)*100/
    # SUM(TR,14)+SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6))
    def tech172(self):
        TR = np.maximum(np.maximum(self.high - self.low, abs(self.high -
                                                             delay(self.close, 1))), abs(self.low - delay(self.close, 1)))
        HD = (self.high - delay(self.high, 1))
        LD = (delay(self.low, 1) - self.low)
        cond1 = (LD > 0) & (LD > HD)
        cond2 = (HD > 0) & (HD > LD)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = LD
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = HD
        tech172 = mean(abs(ts_sum(part1, 14)*100/ts_sum(TR, 14)-ts_sum(part2, 14)*100)/(ts_sum(
            part1, 14)*100/ts_sum(TR, 14)+ts_sum(TR, 14)+ts_sum(part2, 14)*100/ts_sum(TR, 14))*100, 6)
        save_hdf(tech172, 'tech172', self.length)
        return

    # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);
    def tech173(self):
        tech173 = 3*sma(self.close, 13, 2)-2*sma(sma(self.close, 13, 2),
                                                 13, 2)+sma(sma(sma(log(self.close), 13, 2), 13, 2), 13, 2)
        save_hdf(tech173, 'tech173', self.length)
        return

    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def tech174(self):
        cond1 = self.close > delay(self.close, 1)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = stddev(self.close, 20)
        tech174 = sma(part1, 20, 1)
        save_hdf(tech174, 'tech174', self.length)
        return

    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    def tech175(self):
        tech175 = mean(np.maximum(np.maximum((self.high-self.low), abs(
            delay(self.close, 1)-self.high)), abs(delay(self.close, 1)-self.low)), 6)
        save_hdf(tech175, 'tech175', self.length)
        return

    # CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    def tech176(self):
        tech176 = correlation(rank(((self.close - ts_min(self.low, 12)) / (
            ts_max(self.high, 12) - ts_min(self.low, 12)))), rank(self.volume), 6)
        save_hdf(tech176, 'tech176', self.length)
        return

    # ((20-HIGHDAY(HIGH,20))/20)*100
    def tech177(self):
        tech177 = ((20-highday(self.high, 20))/20)*100
        save_hdf(tech177, 'tech177', self.length)
        return

    # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    def tech178(self):
        tech178 = (self.close-delay(self.close, 1)) / \
            delay(self.close, 1)*self.volume
        save_hdf(tech178, 'tech178', self.length)
        return

    # (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    def tech179(self):
        tech179 = (rank(correlation(self.vwap, self.volume, 4)) *
                   rank(correlation(rank(self.low), rank(mean(self.volume, 50)), 12)))
        save_hdf(tech179, 'tech179', self.length)
        return

    # ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))
    def tech180(self):
        cond1 = mean(self.volume, 20) < self.volume
        tech180 = (-1*self.volume)
        tech180[cond1] = (
            (-1 * ts_rank(abs(delta(self.close, 7)), 60)) * sign(delta(self.close, 7)))
        save_hdf(tech180, 'tech180', self.length)
        return

    # SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/
    # SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
    def tech181(self):
        part1 = pd.DataFrame(np.tile(pow(self.benchmark_close-mean(self.benchmark_close, 20), 2),
                                     (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T
        part2 = pd.DataFrame(np.tile(ts_sum(pow(self.benchmark_close-mean(self.benchmark_close, 20), 3), 20),
                                     (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T
        tech181 = ts_sum(((self.close/delay(self.close, 1)-1) -
                          mean((self.close/delay(self.close, 1)-1), 20))-part1, 20)/part2
        save_hdf(tech181, 'tech181', self.length)
        return

    #  #COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    def tech182(self):
        cond1 = (self.close > self.open) & (pd.DataFrame(np.tile((self.benchmark_close > self.benchmark_open),
                                                                 (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T)
        cond2 = (self.close < self.open) & (pd.DataFrame(np.tile((self.benchmark_close < self.benchmark_open),
                                                                 (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1 | cond2] = 1
        tech182 = count(part1, 20)/20
        save_hdf(tech182, 'tech182', self.length)
        return

    # # MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    # def tech183(self):
    #     # tech183 = np.maximum(np.cumsum(self.close-mean(self.close,24)),0)-np.minimum(np.cumsum(self.close-mean(self.close,24)),0)/stddev(self.close,24)
    #     # save_hdf(tech183, 'tech183')
    #     return 0

    # (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    def tech184(self):
        tech184 = (rank(correlation(delay((self.open - self.close), 1),
                                    self.close, 200)) + rank((self.open - self.close)))
        save_hdf(tech184, 'tech184', self.length)
        return

    # RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
    def tech185(self):
        tech185 = rank((-1 * (pow(1 - (self.open / self.close), 2))))
        save_hdf(tech185, 'tech185', self.length)
        return

    # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/
    # (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+
    # DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/
    # (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
    def tech186(self):
        TR = np.maximum(np.maximum(self.high - self.low, abs(self.high -
                                                             delay(self.close, 1))), abs(self.low - delay(self.close, 1)))
        HD = (self.high - delay(self.high, 1))
        LD = (delay(self.low, 1) - self.low)
        cond1 = (LD > 0) & (LD > HD)
        cond2 = (HD > 0) & (HD > LD)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = LD
        part2 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part2[cond2] = HD
        tech186 = ((mean(abs(ts_sum(part1, 14)*100/ts_sum(TR, 14)-ts_sum(part2, 14)*100/ts_sum(TR, 14)) /
                         (ts_sum(part1, 14)*100/ts_sum(TR, 14)+ts_sum(part2, 14)*100/ts_sum(TR, 14))*100, 6) +
                    delay(mean(abs(ts_sum(part1, 14)*100/ts_sum(TR, 14)-ts_sum(part2, 14)*100/ts_sum(TR, 14)) /
                               (ts_sum(part1, 14)*100/ts_sum(TR, 14)+ts_sum(part2, 14)*100/ts_sum(TR, 14))*100, 6), 6))/2)
        save_hdf(tech186, 'tech186', self.length)
        return

    # SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    def tech187(self):
        cond1 = self.open <= delay(self.open, 1)
        tech187 = np.maximum((self.high-self.open),
                             (self.open-delay(self.open, 1)))
        tech187[cond1] = 0
        save_hdf(tech187, 'tech187', self.length)
        return

    # ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    def tech188(self):
        tech188 = ((self.high-self.low-sma(self.high-self.low, 11, 2)
                    )/sma(self.high-self.low, 11, 2))*100
        save_hdf(tech188, 'tech188', self.length)
        return

    # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    def tech189(self):
        tech189 = mean(abs(self.close-mean(self.close, 6)), 6)
        save_hdf(tech189, 'tech189', self.length)
        return

    # LOG((COUNT(CLOSE/DELAY(CLOSE,1)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*
    # (SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<
    # (CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*
    # (SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
    def tech190(self):
        cond1 = (self.close/delay(self.close, 1) -
                 1) > (pow(self.close/delay(self.close, 19), 1/20)-1)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        cond2 = (self.close/delay(self.close, 1) -
                 1) < (pow(self.close/delay(self.close, 19), 1/20)-1)
        part2 = (self.close/delay(self.close, 1)-1) - \
            pow((pow(self.close/delay(self.close, 19), 1/20)-1), 2)
        part2[cond2] = 0
        part3 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part3[cond2] = 1
        part4 = (self.close/delay(self.close, 1)-1) - \
            pow((pow(self.close/delay(self.close, 19), 1/20)-1), 2)
        part4[cond1] = 0
        tech190 = log((count(part1, 20)-1)*(ts_sum(part2, 20)) /
                      ((count(part3, 20))*(ts_sum(part4, 20))))
        save_hdf(tech190, 'tech190', self.length)
        return

    # (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE
    def tech191(self):
        tech191 = (correlation(mean(self.volume, 20), self.low, 5) +
                   ((self.high + self.low) / 2)) - self.close
        save_hdf(tech191, 'tech191', self.length)
        return

    # -1*CORR(VWAP,VOLUME,6)
    def techJLBL(self):
        techJLBL = -1 * correlation(self.vwap, self.volume, 6)
        save_hdf(techJLBL, 'techJLBL', self.length)
        return

    # OPEN/CLOSE
    def techKPQK(self):
        techKPQK = self.open/delay(self.close, 1)
        save_hdf(techKPQK, 'techKPQK', self.length)
        return

    # -1*VOLUME/MEAN(VOLUME,20)
    def techYCCJL(self):
        techYCCJL = -1*self.volume/mean(self.volume, 6)
        save_hdf(techYCCJL, 'techYCCJL', self.length)
        return

    # -1*CORR(HIGH/LOW,VOLUME,6)
    def techLFBL(self):
        techLFBL = -1 * correlation(self.high/self.low, self.volume, 6)
        save_hdf(techLFBL, 'techLFBL', self.length)
        return


def run_func(paras):
    startdate = paras["startdate"]
    enddate = paras["enddate"]
    count = paras["count"]
    length = paras["length"]
    techs = Tech191(startdate, enddate, count, length)
    func_list = paras["func"]
    for func_name in func_list:
        eval("techs." + func_name + "()")
    return


def set_params(func_list, Start_Date, End_Date, count, length):
    td = {"module_name": "Tech191",
          "startdate": Start_Date,
          "enddate": End_Date,
          "count": count,
          "length": length}
    params = []
    for i, sec_code in enumerate(func_list):
        td['func'] = sec_code
        params.append(td.copy())
    return params


if __name__ == '__main__':
    """设置更新日期"""
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    Start_Date = None
    End_Date = str(TradeDate.iloc[-1, 0])
    count = 600  # 用于计算因子的数据长度
    length = TradeDate.shape[0]  # 输出更新的数据长度
    techs = Tech191(Start_Date, End_Date, count, length)
    func_list = [x for x in dir(techs) if x.startswith("t")]
    # func_list.sort(key=lambda x: int(re.sub("\D", "", x)))
    func_list = [func_list[i:i + 1] for i in range(0, len(func_list), 1)]
    paras = set_params(func_list, Start_Date, End_Date, count, length)
    pool = multiprocessing.Pool(12)
    pool.map(run_func, paras)
    pool.close()
    pool.join()
    print("The data of Tech191 update completely!")
