#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : alphafactors.py
# 标准库
import os
import sys
import gc
import multiprocessing

# 第三方库
import numpy as np
import pandas as pd
import tushare as ts
from numpy import abs
from numpy import log
from numpy import sign

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa


class AlphaFactor(object):
    '''
    WorldQuant101个短周期alpha因子
    国泰君安191个短周期技术因子
    '''
    def __init__(self, code, startdate, enddate, count, length):
        """
        [summary]
        Arguments:
            code {[list]} -- [当日股票代码]
            startdate {[str]} -- [数据起始日期]
            enddate {[str]} -- [数据截止日期]
            count {[int]} -- [读取数据的bar数量]
            length {[int]} -- [保存数据时的长度]
        """
        stock_price = get_price(startdate=startdate,
                                enddate=enddate,
                                fields=[
                                    'open_qfq', 'close_qfq', 'low_qfq',
                                    'high_qfq', 'volume', 'amount', 'ret',
                                    'turnover_rate'],
                                count=count)
        benchmark_price = get_price(startdate=startdate,
                                    enddate=enddate,
                                    fields=['index'],
                                    count=count)
        fina_price = get_price(startdate=startdate,
                               enddate=enddate,
                               fields=['pe_ttm', 'eps_FY1', 'cfps', 'cagrgr_PY5', 'cagrpni_PY5',
                                       'sestni_YOY1', 'sestni_YOY3', 'total_assets', 'total_equity',
                                       'total_liab', 'total_mv', 'total_ncl'],
                               count=count)

        self.open = stock_price['open_qfq'].reindex(columns=code).fillna(0)
        self.close = stock_price['close_qfq'].reindex(columns=code).fillna(0)
        self.low = stock_price['low_qfq'].reindex(columns=code).fillna(0)
        self.high = stock_price['high_qfq'].reindex(columns=code).fillna(0)
        self.volume = 100*stock_price['volume'].reindex(columns=code).fillna(0)
        self.amount = 1000*stock_price['amount'].reindex(columns=code).fillna(0)
        self.returns = stock_price['ret'].reindex(columns=code).fillna(0)
        self.benchmark_open = benchmark_price['index']['沪深300open']
        self.benchmark_close = benchmark_price['index']['沪深300close']
        self.vwap = self.amount/(self.volume + 0.001)  # 分母防止被0整除
        self.pre_close = self.close.shift(1).fillna(0)
        #=============================================
        self.turnover_rate = stock_price['turnover_rate'].reindex(columns=code).fillna(0) / 100
        self.benchmark_ret = benchmark_price['index']['沪深300close'].pct_change().fillna(method='ffill').fillna(
            0).replace([np.inf, -np.inf], [0, 0])
        self.pe_ttm = fina_price['pe_ttm'].reindex(columns=code).fillna(0)
        self.eps_FY1 = fina_price['eps_FY1'].reindex(columns=code).fillna(0)
        self.cfps = fina_price['cfps'].reindex(columns=code).fillna(0)
        self.cagrgr_PY5 = fina_price['cagrgr_PY5'].reindex(columns=code).fillna(0)
        self.cagrpni_PY5 = fina_price['cagrpni_PY5'].reindex(columns=code).fillna(0)
        self.sestni_YOY1 = fina_price['sestni_YOY1'].reindex(columns=code).fillna(0)
        self.sestni_YOY3 = fina_price['sestni_YOY3'].reindex(columns=code).fillna(0)
        self.total_assets = fina_price['total_assets'].reindex(columns=code).fillna(0)
        self.total_equity = fina_price['total_equity'].reindex(columns=code).fillna(0)
        self.total_liab = fina_price['total_liab'].reindex(columns=code).fillna(0)
        self.total_ncl = fina_price['total_ncl'].reindex(columns=code).fillna(0)
        self.total_mv = 10000 * fina_price['total_mv'].reindex(columns=code).fillna(0)
        #==============================================
        self.length = length
        self.e = np.ones(self.returns.shape) * np.nan
        #=============================================
        del stock_price, benchmark_price, fina_price
        gc.collect() # 释放内存
        np.seterr(divide='ignore', invalid='ignore')   # 忽略警告
        '''
        1.dataframe与dataframe,0,1比较用np.maximum和np.minimum
          与其他数据比较用ts_max和ts_min，属于时间序列
        2.公式中的幂次方计算一律用pow()函数
        3.benchmark_open与benchmark_close为series
        4.逻辑运算符一律用&, |表示
        '''

    # Alpha#1	 (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha001(self):
        inner = self.close
        inner[self.returns < 0] = stddev(self.returns, 20)
        alpha001 = (rank(ts_argmax(inner**2, 5))-0.5)
        return alpha001.iloc[-self.length]

    # Alpha#2	 (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(self):
        alpha002 = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return alpha002.iloc[-self.length]

    # Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
    def alpha003(self):
        alpha003 = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return alpha003.iloc[-self.length]

    # Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
    def alpha004(self):
        alpha004 = -1 * ts_rank(rank(self.low), 9)
        return alpha004.iloc[-self.length]

    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        alpha005 = (rank((self.open - (ts_sum(self.vwap, 10) / 10))) *
                    (-1 * abs(rank((self.close - self.vwap)))))
        return alpha005.iloc[-self.length]

    # Alpha#6	 (-1 * correlation(open, volume, 10))
    def alpha006(self):
        alpha006 = -1 * correlation(self.open, self.volume, 10)
        return alpha006.iloc[-self.length]

    # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha007(self):
        adv20 = mean(self.volume, 20)
        alpha007 = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(
            delta(self.close, 7))
        alpha007[adv20 >= self.volume] = -1
        return alpha007.iloc[-self.length]

    # Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha008(self):
        alpha008 = -1 * (rank(
            ((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) - delay(
                (ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
        return alpha008.iloc[-self.length]

    # Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha009 = -1 * delta_close
        alpha009[cond_1 | cond_2] = delta_close
        return alpha009.iloc[-self.length]

    # Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha010 = -1 * delta_close
        alpha010[cond_1 | cond_2] = delta_close
        alpha010 = rank(alpha010)
        return alpha010.iloc[-self.length]

    # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    def alpha011(self):
        alpha011 = ((rank(ts_max((self.vwap - self.close), 3)) +
                     rank(ts_min((self.vwap - self.close), 3))) *
                    rank(delta(self.volume, 3)))
        return alpha011.iloc[-self.length]

    # Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha012(self):
        alpha012 = sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))
        return alpha012.iloc[-self.length]

    # Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha013(self):
        alpha013 = -1 * rank(covariance(rank(self.close), rank(self.volume),
                                        5))
        return alpha013.iloc[-self.length]

    # Alpha#14	 ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha014 = -1 * rank(delta(self.returns, 3)) * df
        return alpha014.iloc[-self.length]

    # Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha015 = -1 * ts_sum(rank(df), 3)
        return alpha015.iloc[-self.length]

    # Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha016(self):
        alpha016 = -1 * rank(covariance(rank(self.high), rank(self.volume), 5))
        return alpha016.iloc[-self.length]

    # Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha017(self):
        adv20 = mean(self.volume, 20)
        alpha017 = -1 * (rank(ts_rank(self.close, 10)) *
                         rank(delta(delta(self.close, 1), 1)) *
                         rank(ts_rank((self.volume / adv20), 5)))
        return alpha017.iloc[-self.length]

    # Alpha#18	 (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha018 = -1 * (rank((stddev(abs((self.close - self.open)), 5) +
                               (self.close - self.open)) + df))
        return alpha018.iloc[-self.length]

    # Alpha#19	 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha019(self):
        alpha019 = ((-1 * sign(
            (self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                    (1 + rank(1 + ts_sum(self.returns, 250))))
        return alpha019.iloc[-self.length]

    # Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha020(self):
        alpha020 = -1 * (rank(self.open - delay(self.high, 1)) *
                         rank(self.open - delay(self.close, 1)) *
                         rank(self.open - delay(self.low, 1)))
        return alpha020.iloc[-self.length]

    # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) <
    # ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = mean(self.close, 8) + stddev(self.close, 8) < mean(self.close, 2)
        cond_2 = mean(self.volume, 20) / self.volume < 1
        alpha021 = pd.DataFrame(np.ones_like(self.close),
                                index=self.close.index,
                                columns=self.close.columns)
        #        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
        #                             columns=self.close.columns)
        alpha021[cond_1 | cond_2] = -1
        return alpha021.iloc[-self.length]

    # Alpha#22	 (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        # df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha022 = -1 * delta(df, 5) * rank(stddev(self.close, 20))
        return alpha022.iloc[-self.length]

    # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha023(self):
        cond = (ts_sum(self.high, 20) / 20) < self.high
        alpha023 = pd.DataFrame(np.zeros_like(self.close),
                                index=self.close.index,
                                columns=self.close.columns)
        alpha023[cond] = -1 * delta(self.high, 2)
        return alpha023.iloc[-self.length]

    # Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100),
    #  100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha024(self):
        cond = delta(mean(self.close, 100), 100) / delay(self.close,100) <= 0.05
        alpha024 = -1 * delta(self.close, 3)
        alpha024[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha024.iloc[-self.length]

    # Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha025(self):
        adv20 = mean(self.volume, 20)
        alpha025 = rank(((((-1 * self.returns) * adv20) * self.vwap) *
                         (self.high - self.close)))
        return alpha025.iloc[-self.length]

    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha026 = -1 * ts_max(df, 3)
        return alpha026.iloc[-self.length]

    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    #可能存在问题，我自己的数据测试了很多次值全为1，可能需要调整6,2这些参数？
    def alpha027(self):
        alpha027 = rank(
            (mean(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha027[alpha027 > 0.5] = -1
        alpha027[alpha027 <= 0.5] = 1
        return alpha027.iloc[-self.length]

    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = mean(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha028 = scale(((df + ((self.high + self.low) / 2)) - self.close))
        return alpha028.iloc[-self.length]

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),
    # 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        alpha029 = (ts_min(
            rank(
                rank(
                    scale(
                        log(
                            ts_sum(
                                rank(
                                    rank(-1 * rank(delta(
                                        (self.close - 1), 5)))), 2))))), 5) +
                    ts_rank(delay((-1 * self.returns), 6), 5))
        return alpha029.iloc[-self.length]

    # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2))))
    # +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(
            delay(delta_close, 2))
        alpha030 = ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(
            self.volume, 20)
        return alpha030.iloc[-self.length]

    # Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) +
    # rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        adv20 = mean(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf],
                                                      0).fillna(value=0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))),10))))
        p2 = rank((-1 * delta(self.close, 3)))
        p3 = sign(scale(df))
        alpha031 = p1 + p2 + p3
        return alpha031.iloc[-self.length]

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha032(self):
        alpha032 = scale(((mean(self.close, 7) / 7) - self.close)) + (
            20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))
        return alpha032.iloc[-self.length]

    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        alpha033 = rank(-1 + (self.open / self.close))
        return alpha033.iloc[-self.length]

    # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        alpha034 = rank(2 - rank(inner) - rank(delta(self.close, 1)))
        return alpha034.iloc[-self.length]

    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        alpha035 = ((ts_rank(self.volume, 32) *
                     (1 - ts_rank(self.close + self.high - self.low, 16))) *
                    (1 - ts_rank(self.returns, 32)))
        return alpha035.iloc[-self.length]

    # Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) +
    # (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha036(self):
        adv20 = mean(self.volume, 20)
        alpha036 = (((((2.21 * rank(
            correlation(
                (self.close - self.open), delay(self.volume, 1), 15))) +
                       (0.7 * rank((self.open - self.close)))) +
                      (0.73 * rank(ts_rank(delay(
                          (-1 * self.returns), 6), 5)))) +
                     rank(abs(correlation(self.vwap, adv20, 6)))) +
                    (0.6 * rank((((mean(self.close, 200) / 200) - self.open) *
                                 (self.close - self.open)))))
        return alpha036.iloc[-self.length]

    # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        alpha037 = rank(
            correlation(delay(self.open - self.close, 1), self.close,
                        200)) + rank(self.open - self.close)
        return alpha037.iloc[-self.length]

    # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        alpha038 = -1 * rank(ts_rank(self.open, 10)) * rank(inner)
        return alpha038.iloc[-self.length]

    # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        adv20 = mean(self.volume, 20)
        alpha039 = ((-1 * rank(
            delta(self.close, 7) *
            (1 - rank(decay_linear((self.volume / adv20), 9))))) *
                    (1 + rank(mean(self.returns, 250))))
        return alpha039.iloc[-self.length]

    # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        alpha040 = -1 * rank(stddev(self.high, 10)) * correlation(
            self.high, self.volume, 10)
        return alpha040.iloc[-self.length]

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        alpha041 = pow((self.high * self.low), 0.5) - self.vwap
        return alpha041.iloc[-self.length]

    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        alpha042 = rank((self.vwap - self.close)) / rank(
            (self.vwap + self.close))
        return alpha042.iloc[-self.length]

    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self):
        adv20 = mean(self.volume, 20)
        alpha043 = ts_rank(self.volume / adv20, 20) * ts_rank(
            (-1 * delta(self.close, 7)), 8)
        return alpha043.iloc[-self.length]

    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        alpha044 = -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha044.iloc[-self.length]

    # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * 
    # rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha045 = -1 * (rank(mean(delay(self.close, 5), 20)) * df * rank(
            correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))
        return alpha045.iloc[-self.length]

    # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) :
    # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
            (delay(self.close, 10) - self.close) / 10)
        alpha046 = (-1 * delta(self.close))
        alpha046[inner < 0] = 1
        alpha046[inner > 0.25] = -1
        return alpha046.iloc[-self.length]

    # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha047(self):
        adv20 = mean(self.volume, 20)
        alpha047 = ((((rank(
            (1 / self.close)) * self.volume) / adv20) * ((self.high * rank(
                (self.high - self.close))) / (mean(self.high, 5) / 5))) - rank(
                    (self.vwap - delay(self.vwap, 5))))

        return alpha047.iloc[-self.length]

    # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))

    # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) -
                 ((delay(self.close, 10) - self.close) / 10))
        alpha049 = (-1 * delta(self.close))
        alpha049[inner < -0.1] = 1
        return alpha049.iloc[-self.length]

    # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha050(self):
        alpha050 = (-1 * ts_max(
            rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))
        return alpha050.iloc[-self.length]

    # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) -
                 ((delay(self.close, 10) - self.close) / 10))
        alpha051 = (-1 * delta(self.close))
        alpha051[inner < -0.05] = 1
        return alpha051.iloc[-self.length]

    # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha052(self):
        alpha052 = (((-1 * delta(ts_min(self.low, 5), 5)) * rank(
            ((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) *
                    ts_rank(self.volume, 5))
        return alpha052.iloc[-self.length]

    # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        alpha053 = -1 * delta(
            (((self.close - self.low) - (self.high - self.close)) / inner), 9)
        return alpha053.iloc[-self.length]

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        alpha054 = -1 * (self.low - self.close) * (self.open**
                                                   5) / (inner *
                                                         (self.close**5))
        return alpha054.iloc[-self.length]

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(
            0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        alpha055 = -1 * correlation(rank(inner), rank(self.volume), 6)
        return alpha055.iloc[-self.length]

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    #本Alpha使用了cap|市值，暂未取到该值
    #    def alpha056(self):
    #        return (0 - (1 * (rank((mean(self.returns, 10) / mean(mean(self.returns, 2), 3))) * rank((self.returns * self.cap)))))

    # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    def alpha057(self):
        alpha057 = (0 - (1 *((self.close - self.vwap) /
                    decay_linear(rank(ts_argmax(self.close, 30)), 2))))
        return alpha057.iloc[-self.length]

    # Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))

    # Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))

    # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) -
                 (self.high - self.close)) * self.volume / divisor
        alpha060 = -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
        return alpha060.iloc[-self.length]

    # Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha061(self):
        adv180 = mean(self.volume, 180)
        alpha061 = ((rank((self.vwap - ts_min(self.vwap, 16))) < rank(
            correlation(self.vwap, adv180, 18)))*1)
        return alpha061.iloc[-self.length]

    # Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha062(self):
        adv20 = mean(self.volume, 20)
        alpha062 = ((rank(correlation(self.vwap, mean(adv20, 22), 10)) < rank(
            ((rank(self.open) + rank(self.open)) < (rank(
                ((self.high + self.low) / 2)) + rank(self.high))))) * -1)
        return alpha062.iloc[-self.length]

    # Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)

    # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) <
    # rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha064(self):
        adv120 = mean(self.volume, 120)
        alpha064 = ((rank(
            correlation(
                mean(((self.open * 0.178404) +
                     (self.low *
                      (1 - 0.178404))), 13), mean(adv120, 13), 17)) < rank(
                          delta(
                              ((((self.high + self.low) / 2) * 0.178404) +
                               (self.vwap * (1 - 0.178404))), 3.69741))) * -1)
        return alpha064.iloc[-self.length]

    # Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    def alpha065(self):
        adv60 = mean(self.volume, 60)
        alpha065 = ((rank(
            correlation(
                ((self.open * 0.00817205) +
                 (self.vwap * (1 - 0.00817205))), mean(adv60, 9), 6)) < rank(
                     (self.open - ts_min(self.open, 14)))) * -1)
        return alpha065.iloc[-self.length]

    # Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) +
    # (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha066(self):
        alpha066 = ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(
            decay_linear(
                ((((self.low * 0.96633) + (self.low *
                                           (1 - 0.96633))) - self.vwap) /
                 (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
        return alpha066.iloc[-self.length]

    # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)

    # Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    def alpha068(self):
        adv15 = mean(self.volume, 15)
        alpha068 = (
            (ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) < rank(
                delta(((self.close * 0.518371) +
                       (self.low * (1 - 0.518371))), 1.06157))) * -1)
        return alpha068.iloc[-self.length]

    # Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)

    # Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

    # Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501),
    # 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    def alpha071(self):
        adv180 = mean(self.volume, 180)
        p1 = ts_rank(
            decay_linear(
                correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18),
                4), 16)
        p2 = ts_rank(
            decay_linear((rank(
                ((self.low + self.open) - (self.vwap + self.vwap))).pow(2)),
                         16), 4)
        alpha071 = pd.DataFrame(np.maximum(p1, p2),
                                index=self.close.index,
                                columns=self.close.columns)
        return alpha071.iloc[-self.length]
        #return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))

    # Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    def alpha072(self):
        adv40 = mean(self.volume, 40)
        alpha072 = (rank(
            decay_linear(correlation(
                ((self.high + self.low) / 2), adv40, 9), 10)) / rank(
                    decay_linear(
                        correlation(ts_rank(self.vwap, 4),
                                    ts_rank(self.volume, 19), 7), 3)))
        return alpha072.iloc[-self.length]

    # Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) +
    # (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    def alpha073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        p2 = ts_rank(
            decay_linear(((delta(
                ((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) /
                           ((self.open * 0.147155) +
                            (self.low * (1 - 0.147155)))) * -1), 3), 17)
        alpha073 = (-1 * pd.DataFrame(np.maximum(p1, p2),
                                      index=self.close.index,
                                      columns=self.close.columns))
        return alpha073.iloc[-self.length]
        #return (max(rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE),ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)) * -1)

    # Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    def alpha074(self):
        adv30 = mean(self.volume, 30)
        alpha074 = ((rank(correlation(self.close, mean(adv30, 37), 15)) < rank(
            correlation(
                rank(((self.high * 0.0261661) +
                      (self.vwap *
                       (1 - 0.0261661)))), rank(self.volume), 11))) * -1)
        return alpha074.iloc[-self.length]

    # Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
    def alpha075(self):
        adv50 = mean(self.volume, 50)
        alpha075 = ((rank(correlation(self.vwap, self.volume, 4)) < rank(
            correlation(rank(self.low), rank(adv50), 12))) * 1)
        return alpha075.iloc[-self.length]

    # Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)

    # Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    def alpha077(self):
        adv40 = mean(self.volume, 40)
        p1 = rank(
            decay_linear(((((self.high + self.low) / 2) + self.high) -
                          (self.vwap + self.high)), 20))
        p2 = rank(
            decay_linear(correlation(((self.high + self.low) / 2), adv40, 3),
                         6))
        alpha077 = (pd.DataFrame(np.minimum(p1, p2),
                                 index=self.close.index,
                                 columns=self.close.columns))
        return alpha077.iloc[-self.length]
        #return min(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE),rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE))

    # Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    def alpha078(self):
        adv40 = mean(self.volume, 40)
        alpha078 = (rank(
            correlation(
                ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))),
                       20), ts_sum(adv40, 20),
                7)).pow(
                    rank(correlation(rank(self.vwap), rank(self.volume), 6))))
        return alpha078.iloc[-self.length]

    # Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))

    # Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

    # Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    def alpha081(self):
        adv10 = mean(self.volume, 10)
        alpha081 = ((rank(log(product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50),8)).pow(4))), 15))) < rank(correlation(rank(self.vwap), rank(self.volume), 5))) * -1)
        return alpha081.iloc[-self.length]

    # Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

    # Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    def alpha083(self):
        alpha083 = ((rank(
            delay(
                ((self.high - self.low) /
                 (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) /
                    (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) /
                     (self.vwap - self.close)))
        return alpha083.iloc[-self.length]

    # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    def alpha084(self):
        alpha084 = pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21),delta(self.close, 5))
        return alpha084.iloc[-self.length]

    # Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    def alpha085(self):
        adv30 = mean(self.volume, 30)
        alpha085 = (rank(
            correlation(
                ((self.high * 0.876703) + (self.close * (1 - 0.876703))),
                adv30, 10)).pow(
                    rank(
                        correlation(ts_rank(((self.high + self.low) / 2), 4),
                                    ts_rank(self.volume, 10), 7))))
        return alpha085.iloc[-self.length]

    # Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    def alpha086(self):
        adv20 = mean(self.volume, 20)
        alpha086 = ((ts_rank(correlation(self.close, mean(adv20, 15), 6), 20) < rank(((self.open + self.close) - (self.vwap + self.open)))) * -1)
        return alpha086.iloc[-self.length]

    # Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)

    # Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    def alpha088(self):
        adv60 = mean(self.volume, 60)
        p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) -(rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8), 7),3)
        alpha088 = (pd.DataFrame(np.minimum(p1, p2),index=self.close.index,columns=self.close.columns))
        return alpha088.iloc[-self.length]
        #return min(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),8).CLOSE), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,20.6966), 8).to_frame(), 7).CLOSE, 3))

    # Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))

    # Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)

    # Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)

    # Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    def alpha092(self):
        adv30 = mean(self.volume, 30)
        p1 = ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) <(self.low + self.open)), 15), 19)
        p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8), 7), 7)
        alpha092 = (pd.DataFrame(np.minimum(p1, p2),index=self.close.index,columns=self.close.columns))
        return alpha092.iloc[-self.length]
        #return  min(ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,19), ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE,7))

    # Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))

    # Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    def alpha094(self):
        adv60 = mean(self.volume, 60)
        alpha094 = ((rank((self.vwap - ts_min(self.vwap, 12))).pow(ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18),3)) * -1))
        return alpha094.iloc[-self.length]

    # Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    def alpha095(self):
        adv40 = mean(self.volume, 40)
        alpha095 = ((rank((self.open - ts_min(self.open, 12))) < ts_rank((rank(correlation(mean(((self.high + self.low) / 2), 19), mean(adv40, 19),13)).pow(5)), 12))*1)
        return alpha095.iloc[-self.length]

    # Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    def alpha096(self):
        adv60 = mean(self.volume, 60)
        p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4),4), 8)
        p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4),13), 14), 13)
        alpha096 = (-1 * pd.DataFrame(np.maximum(p1, p2),index=self.close.index,columns=self.close.columns))
        return alpha096.iloc[-self.length]
        # return (max(ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4),4).CLOSE, 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)) * -1)

    # Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

    # Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    def alpha098(self):
        adv5 = mean(self.volume, 5)
        adv15 = mean(self.volume, 15)
        alpha098 = (rank(decay_linear(correlation(self.vwap, mean(adv5, 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9),7), 8)))
        return alpha098.iloc[-self.length]

    # Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    def alpha099(self):
        adv60 = mean(self.volume, 60)
        alpha099 = ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <rank(correlation(self.low, self.volume, 6))) * -1)
        return alpha099.iloc[-self.length]

    # Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))

    # Alpha#101	 ((close - open) / ((high - low) + .001))
    def alpha101(self):
        alpha101 = (self.close - self.open) / ((self.high - self.low) + 0.001)
        return alpha101.iloc[-self.length]

    '''==================================================================================
                          国泰君安191个短周期alpha因子从这里开始
    =================================================================================='''
        # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    def tech001(self):
        tech001 = (-1*correlation(rank(delta(log(self.volume), 1)),
                                  rank(((self.close-self.open)/self.open)), 6))
        return tech001.iloc[-self.length]

    # (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def tech002(self):
        tech002 = (-1*delta((((self.close-self.low) -
                              (self.high-self.close))/(self.high-self.low)), 1))
        return tech002.iloc[-self.length]

    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    def tech003(self):
        part1 = self.close
        part1[self.close == delay(self.close, 1)] = 0
        part2 = np.maximum(self.high, delay(self.close, 1))
        part2[self.close > delay(self.close, 1)] = np.minimum(
            self.low, delay(self.close, 1))
        tech003 = pd.DataFrame(
            ts_sum(part1-part2, 6), index=self.close.index, columns=self.close.columns)
        return tech003.iloc[-self.length]

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
        return tech004.iloc[-self.length]

    # (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    def tech005(self):
        tech005 = (-1*ts_max(correlation(ts_rank(self.volume, 5),
                                         ts_rank(self.high, 5), 5), 5))
        return tech005.iloc[-self.length]

    # (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    def tech006(self):
        tech006 = (rank(sign(delta((self.open*0.85+self.high*0.15), 4)))*(-1))
        return tech006.iloc[-self.length]

    # ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def tech007(self):
        tech007 = ((rank(ts_max((self.vwap-self.close), 3)) +
                    rank(ts_min((self.vwap-self.close), 3))) *
                   rank(delta(self.volume, 3)))
        return tech007.iloc[-self.length]

    # RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    def tech008(self):
        tech008 = rank(
            delta(((((self.high+self.low)/2)*0.2)+(self.vwap*0.8)), 4)*(-1))
        return tech008.iloc[-self.length]

    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    def tech009(self):
        tech009 = sma(((self.high+self.low)/2-(delay(self.high, 1) +
                                               delay(self.low, 1))/2)*(self.high-self.low)/self.volume, 7, 2)
        return tech009.iloc[-self.length]

    # (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    def tech010(self):
        part1 = self.returns
        part1[self.returns < 0] = stddev(self.returns, 20)
        tech010 = rank(ts_max(pow(part1, 2), 5))
        return tech010.iloc[-self.length]

    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    def tech011(self):
        tech011 = ts_sum(((self.close-self.low)-(self.high -
                                                 self.close))/(self.high-self.low)*self.volume, 6)
        return tech011.iloc[-self.length]

    # (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    def tech012(self):
        tech012 = (rank((self.open - (ts_sum(self.vwap, 10) / 10)))
                   ) * (-1 * (rank(abs((self.close - self.vwap)))))
        return tech012.iloc[-self.length]

    # (((HIGH * LOW)^0.5) - VWAP)
    def tech013(self):
        tech013 = (pow((self.high * self.low), 0.5) - self.vwap)
        return tech013.iloc[-self.length]

    # CLOSE-DELAY(CLOSE,5)
    def tech014(self):
        tech014 = self.close-delay(self.close, 5)
        return tech014.iloc[-self.length]

    # OPEN/DELAY(CLOSE,1)-1
    def tech015(self):
        tech015 = self.open/delay(self.close, 1)-1
        return tech015.iloc[-self.length]

    # (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    def tech016(self):
        tech016 = (-1*ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))
        return tech016.iloc[-self.length]

    # RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
    def tech017(self):
        tech017 = pow(rank((self.vwap - ts_max(self.vwap, 15))),
                      delta(self.close, 5))
        return tech017.iloc[-self.length]

    # CLOSE/DELAY(CLOSE,5)
    def tech018(self):
        tech018 = self.close/delay(self.close, 5)
        return tech018.iloc[-self.length]

    # (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
    def tech019(self):
        cond1 = (self.close < delay(self.close, 5))
        cond2 = (self.close == delay(self.close, 5))
        tech019 = (self.close-delay(self.close, 5))/self.close
        tech019[cond1] = (self.close-delay(self.close, 5))/delay(self.close, 5)
        tech019[cond2] = 0
        return tech019.iloc[-self.length]

    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    def tech020(self):
        tech020 = (self.close-delay(self.close, 6))/delay(self.close, 6)*100
        return tech020.iloc[-self.length]

    # REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    def tech021(self):
        tech021 = regbeta(mean(self.close, 6), sequence(6), 6)
        return tech021.iloc[-self.length]

    # SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def tech022(self):
        tech022 = sma(((self.close-mean(self.close, 6))/mean(self.close, 6) -
                       delay((self.close-mean(self.close, 6))/mean(self.close, 6), 3)), 12, 1)
        return tech022.iloc[-self.length]

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
        return tech023.iloc[-self.length]

    # SMA(CLOSE-DELAY(CLOSE,5),5,1)
    def tech024(self):
        tech024 = sma(self.close-delay(self.close, 5), 5, 1)
        return tech024.iloc[-self.length]

    # ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))
    def tech025(self):
        tech025 = ((-1 * rank((delta(self.close, 7) * (1 - rank(decay_linear((self.volume /
                                                                              mean(self.volume, 20)), 9)))))) * (1 + rank(ts_sum(self.returns, 250))))
        return tech025.iloc[-self.length]

    # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
    def tech026(self):
        tech026 = ((((ts_sum(self.close, 7) / 7) - self.close)) +
                   ((correlation(self.vwap, delay(self.close, 5), 230))))
        return tech026.iloc[-self.length]

    # WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    def tech027(self):
        tech027 = wma((self.close-delay(self.close, 3))/delay(self.close, 3)
                      * 100+(self.close-delay(self.close, 6))/delay(self.close, 6)*100, 12)
        return tech027.iloc[-self.length]

    # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/( MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    def tech028(self):
        tech028 = 3*sma((self.close-ts_min(self.low, 9))/(ts_max(self.high, 9)-ts_min(self.low, 9))*100, 3, 1) - \
            2*sma(sma((self.close-ts_min(self.low, 9)) /
                      (ts_max(self.high, 9)-ts_max(self.low, 9))*100, 3, 1), 3, 1)
        return tech028.iloc[-self.length]

    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def tech029(self):
        tech029 = (self.close-delay(self.close, 6)) / \
            delay(self.close, 6)*self.volume
        return tech029.iloc[-self.length]

    # # WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML，60))^2,20)
    # def tech030(self):
    #     return 0

    # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    def tech031(self):
        tech031 = (self.close-mean(self.close, 12))/mean(self.close, 12)*100
        return tech031.iloc[-self.length]

    # (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    def tech032(self):
        tech032 = (-1*ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3))
        return tech032.iloc[-self.length]

    # ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))
    def tech033(self):
        tech033 = ((((-1*ts_min(self.low, 5))+delay(ts_min(self.low, 5), 5))*rank(
            ((ts_sum(self.returns, 240)-ts_sum(self.returns, 20))/220)))*ts_rank(self.volume, 5))
        return tech033.iloc[-self.length]

    # MEAN(CLOSE,12)/CLOSE
    def tech034(self):
        tech034 = mean(self.close, 12)/self.close
        return tech034.iloc[-self.length]

    # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)
    def tech035(self):
        part1 = (np.minimum(rank(decay_linear(delta(self.open, 1), 15)), rank(decay_linear(
            correlation((self.volume), ((self.open * 0.65) + (self.open * 0.35)), 17), 7))) * -1)
        tech035 = pd.DataFrame(
            part1, index=self.close.index, columns=self.close.columns)
        return tech035.iloc[-self.length]

    # RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP))6, 2))
    def tech036(self):
        tech036 = rank(
            ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2))
        return tech036.iloc[-self.length]

    # (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    def tech037(self):
        tech037 = (-1*rank(((ts_sum(self.open, 5)*ts_sum(self.returns, 5)) -
                            delay((ts_sum(self.open, 5)*ts_sum(self.returns, 5)), 10))))
        return tech037.iloc[-self.length]

    # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    def tech038(self):
        cond1 = ((ts_sum(self.high, 20)/20) < self.high)
        tech038 = pd.DataFrame(np.zeros(self.close.shape),
                               index=self.close.index, columns=self.close.columns)
        tech038[cond1] = (-1*delta(self.high, 2))
        return tech038.iloc[-self.length]

    # ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
    def tech039(self):
        tech039 = ((rank(decay_linear(delta((self.close), 2), 8)) - rank(decay_linear(correlation(
            ((self.vwap * 0.3) + (self.open * 0.7)), ts_sum(mean(self.vwap, 180), 37), 14), 12))) * -1)
        return tech039.iloc[-self.length]

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
        return tech040.iloc[-self.length]

    # (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
    def tech041(self):
        tech041 = (rank(ts_max(delta((self.vwap), 3), 5)) * -1)
        return tech041.iloc[-self.length]

    # ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
    def tech042(self):
        tech042 = ((-1 * rank(stddev(self.high, 10))) *
                   correlation(self.high, self.volume, 10))
        return tech042.iloc[-self.length]

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
    def tech043(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close < delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part1[cond2] = -self.volume
        tech043 = ts_sum(part1, 6)
        return tech043.iloc[-self.length]

    # (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
    def tech044(self):
        tech044 = (ts_rank(decay_linear(correlation(((self.low)), mean(
            self.volume, 10), 7), 6), 4) + ts_rank(decay_linear(delta((self.vwap), 3), 10), 15))
        return tech044.iloc[-self.length]

    # (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
    def tech045(self):
        tech045 = (rank(delta((((self.close * 0.6) + (self.open * 0.4))), 1))
                   * rank(correlation(self.vwap, mean(self.volume, 150), 15)))
        return tech045.iloc[-self.length]

    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    def tech046(self):
        tech046 = (mean(self.close, 3)+mean(self.close, 6) +
                   mean(self.close, 12)+mean(self.close, 24))/(4*self.close)
        return tech046.iloc[-self.length]

    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    def tech047(self):
        tech047 = sma((ts_max(self.high, 6) - self.close) /
                      (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9, 1)
        return tech047.iloc[-self.length]

    # (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2))))
    # + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
    def tech048(self):
        tech048 = (-1*((ts_rank(((sign((self.close - delay(self.close, 1))) + sign((delay(self.close, 1)
                                                                                    - delay(self.close, 2)))) + sign((delay(self.close, 2) - delay(self.close, 3)))))) *
                       ts_sum(self.volume, 5)) / ts_sum(self.volume, 20))
        return tech048.iloc[-self.length]

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
        return tech049.iloc[-self.length]

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
        return tech050.iloc[-self.length]

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
        return tech051.iloc[-self.length]

    # SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3-LOW,1)),26)* 100
    def tech052(self):
        tech052 = ts_sum(np.maximum(0, self.high-delay((self.high+self.low+self.close)/3, 1)), 26) / \
            ts_sum(np.maximum(
                0, delay((self.high+self.low+self.close)/3-self.low, 1)), 26)*100
        return tech052.iloc[-self.length]

    # COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    def tech053(self):
        cond1 = (self.close > delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        tech053 = count(part1, 12) / 12 * 100
        return tech053.iloc[-self.length]

    # (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    def tech054(self):
        tech054 = (-1*rank((stddev(abs(self.close-self.open)) +
                            (self.close-self.open))+correlation(self.close, self.open, 10)))
        return tech054.iloc[-self.length]

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
        return tech055.iloc[-self.length]

    # (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))
    def tech056(self):
        tech056 = ((rank((self.open - ts_min(self.open, 12))) <
                    rank(pow(rank(correlation(ts_sum(((self.high + self.low) / 2), 19), ts_sum(mean(self.volume, 40), 19), 13)), 5)))*1)
        return tech056.iloc[-self.length]

    # SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    def tech057(self):
        tech057 = sma((self.close-ts_min(self.low, 9)) /
                      (ts_max(self.high, 9))*100, 3, 1)
        return tech057.iloc[-self.length]

    # COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    def tech058(self):
        cond1 = (self.close > delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        tech058 = count(part1, 20)/20*100
        return tech058.iloc[-self.length]

    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,D ELAY(CLOSE,1)))),20)
    def tech059(self):
        cond1 = (self.close == delay(self.close, 1))
        cond2 = (self.close > delay(self.close, 1))
        part1 = self.close
        part1[cond1] = 0
        part2 = np.maximum(self.high, delay(self.close, 1))
        part2[cond2] = np.minimum(self.low, delay(self.close, 1))
        tech059 = ts_sum(part1-part2, 20)
        return tech059.iloc[-self.length]

    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
    def tech060(self):
        tech060 = ts_sum(((self.close-self.low)-(self.high -
                                                 self.close))/(self.high-self.low)*self.volume, 20)
        return tech060.iloc[-self.length]

    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,80), 8)), 17))) * -1)
    def tech061(self):
        tech061 = (np.maximum(rank(decay_linear(delta(self.vwap, 1), 12)), rank(
            decay_linear(rank(correlation(self.low, mean(self.volume, 80), 8)), 17))) * -1)
        return tech061.iloc[-self.length]

    # (-1 * CORR(HIGH, RANK(VOLUME), 5))
    def tech062(self):
        tech062 = (-1*correlation(self.high, rank(self.volume), 5))
        return tech062.iloc[-self.length]

    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    def tech063(self):
        tech063 = sma(np.maximum(self.close-delay(self.close, 1), 0),
                      6, 1)/sma(abs(self.close-delay(self.close, 1)), 6, 1)*100
        return tech063.iloc[-self.length]

    # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,64)), 4), 13), 14))) * -1)
    def tech064(self):
        tech064 = (np.maximum(rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4)), rank(
            decay_linear(ts_max(correlation(rank(self.close), rank(mean(self.volume, 64)), 4), 13), 14))) * -1)
        return tech064.iloc[-self.length]

    # MEAN(CLOSE,6)/CLOSE
    def tech065(self):
        tech065 = mean(self.close, 6)/self.close
        return tech065.iloc[-self.length]

    # (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    def tech066(self):
        tech066 = (self.close-mean(self.close, 6))/mean(self.close, 6)*100
        return tech066.iloc[-self.length]

    ##################################################################
    def tech067(self):
        tech067 = sma(np.maximum(self.close-delay(self.close, 1), 0),
                      24, 1)/sma(abs(self.close-delay(self.close, 1)), 24, 1)*100
        return tech067.iloc[-self.length]

    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    def tech068(self):
        tech068 = sma(((self.high+self.low)/2-(delay(self.high, 1) +
                                               delay(self.low, 1))/2)*(self.high-self.low)/self.volume, 15, 2)
        return tech068.iloc[-self.length]

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
        return tech069.iloc[-self.length]

    # STD(AMOUNT, 6)
    def tech070(self):
        tech070 = stddev(self.amount, 6)
        return tech070.iloc[-self.length]

    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    def tech071(self):
        tech071 = (self.close-mean(self.close, 24))/mean(self.close, 24)*100
        return tech071.iloc[-self.length]

    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    def tech072(self):
        tech072 = sma((ts_max(self.high, 6)-self.close) /
                      (ts_max(self.high, 6)-ts_min(self.low, 6))*100, 15, 1)
        return tech072.iloc[-self.length]

    # ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
    def tech073(self):
        tech073 = ((ts_rank(decay_linear(decay_linear(correlation((self.close), self.volume, 10), 16), 4),
                            5) - rank(decay_linear(correlation(self.vwap, mean(self.volume, 30), 4), 3))) * -1)
        return tech073.iloc[-self.length]

    # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
    def tech074(self):
        tech074 = (rank(correlation(ts_sum(((self.low * 0.35) + (self.vwap * 0.65)), 20), ts_sum(
            mean(self.volume, 40), 20), 7)) + rank(correlation(rank(self.vwap), rank(self.volume), 6)))
        return tech074.iloc[-self.length]

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
        return tech075.iloc[-self.length]

    # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    def tech076(self):
        tech076 = stddev(abs((self.close/delay(self.close, 1)-1))/self.volume, 20) / \
            mean(abs((self.close/delay(self.close, 1)-1))/self.volume, 20)
        return tech076.iloc[-self.length]

    # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6))
    def tech077(self):
        tech077 = np.minimum(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)),
                             rank(decay_linear(correlation(((self.high + self.low) / 2), mean(self.volume, 40), 3), 6)))
        return tech077.iloc[-self.length]

    # ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    def tech078(self):
        tech078 = ((self.high+self.low+self.close)/3-mean((self.high+self.low+self.close)/3, 12)) / \
            (0.015*mean(abs(self.close-mean((self.high+self.low+self.close)/3, 12)), 12))
        return tech078.iloc[-self.length]

    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    def tech079(self):
        tech079 = sma(np.maximum(self.close-delay(self.close, 1), 0),
                      12, 1)/sma(abs(self.close-delay(self.close, 1)), 12, 1)*100
        return tech079.iloc[-self.length]

    # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    def tech080(self):
        tech080 = (self.volume-delay(self.volume, 5))/delay(self.volume, 5)*100
        return tech080.iloc[-self.length]

    # SMA(VOLUME,21,2)
    def tech081(self):
        tech081 = sma(self.volume, 21, 2)
        return tech081.iloc[-self.length]

    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    def tech082(self):
        tech082 = sma((ts_max(self.high, 6)-self.close) /
                      (ts_max(self.high, 6)-ts_min(self.low, 6))*100, 20, 1)
        return tech082.iloc[-self.length]

    # (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    def tech083(self):
        tech083 = (-1 * rank(covariance(rank(self.high), rank(self.volume), 5)))
        return tech083.iloc[-self.length]

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    def tech084(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close < delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part1[cond2] = -self.volume
        tech084 = ts_sum(part1, 20)
        return tech084.iloc[-self.length]

    # (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
    def tech085(self):
        tech085 = (ts_rank((self.volume / mean(self.volume, 20)), 20)
                   * ts_rank((-1 * delta(self.close, 7)), 8))
        return tech085.iloc[-self.length]

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
        return tech086.iloc[-self.length]

    # ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
    def tech087(self):
        tech087 = ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(((((self.low * 0.9) +
                    (self.low * 0.1)) - self.vwap) / (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
        return tech087.iloc[-self.length]

    # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100

    def tech088(self):
        tech088 = (self.close-delay(self.close, 20))/delay(self.close, 20)*100
        return tech088.iloc[-self.length]

    # 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    def tech089(self):
        tech089 = 2*(sma(self.close, 13, 2)-sma(self.close, 27, 2) -
                     sma(sma(self.close, 13, 2)-sma(self.close, 27, 2), 10, 2))
        return tech089.iloc[-self.length]

    # ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
    def tech090(self):
        tech090 = (rank(correlation(rank(self.vwap), rank(self.volume), 5)) * -1)
        return tech090.iloc[-self.length]

    # ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
    def tech091(self):
        tech091 = ((rank((self.close - ts_max(self.close, 5))) *
                    rank(correlation((mean(self.volume, 40)), self.low, 5))) * -1)
        return tech091.iloc[-self.length]

    # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
    def tech092(self):
        tech092 = (np.maximum(rank(decay_linear(delta(((self.close*0.35)+(self.vwap*0.65)), 2), 3)),
                              ts_rank(decay_linear(abs(correlation((mean(self.volume, 180)), self.close, 13)), 5), 15))*-1)
        return tech092.iloc[-self.length]

    # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    def tech093(self):
        cond1 = (self.open >= delay(self.open, 1))
        part1 = np.maximum((self.open-self.low),
                           (self.open-delay(self.open, 1)))
        part1[cond1] = 0
        tech093 = ts_sum(part1, 20)
        return tech093.iloc[-self.length]

    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    def tech094(self):
        cond1 = (self.close > delay(self.close, 1))
        cond2 = (self.close < delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part1[cond2] = -self.volume
        tech094 = ts_sum(part1, 30)
        return tech094.iloc[-self.length]

    # STD(AMOUNT,20)
    def tech095(self):
        tech095 = stddev(self.amount, 20)
        return tech095.iloc[-self.length]
    # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)

    def tech096(self):
        tech096 = sma(sma((self.close-ts_min(self.low, 9)) /
                          (ts_max(self.high, 9)-ts_min(self.low, 9))*100, 3, 1), 3, 1)
        return tech096.iloc[-self.length]

    # STD(VOLUME,10)
    def tech097(self):
        tech097 = stddev(self.volume, 10)
        return tech097.iloc[-self.length]

    # ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
    def tech098(self):
        cond1 = ((delta((ts_sum(self.close, 100)/100), 100) /
                  delta(self.close, 100)) <= 0.05)
        tech098 = (-1*delta(self.close, 3))
        tech098[cond1] = (-1 * (self.close - ts_min(self.close, 100)))
        return tech098.iloc[-self.length]

    # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    def tech099(self):
        tech099 = (-1 * rank(covariance(rank(self.close), rank(self.volume), 5)))
        return tech099.iloc[-self.length]

    # STD(VOLUME,20)
    def tech100(self):
        tech100 = stddev(self.volume, 20)
        return tech100.iloc[-self.length]

    # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1)
    def tech101(self):
        tech101 = ((rank(correlation(self.close, ts_sum(mean(self.volume, 30), 37), 15)) < rank(
            correlation(rank(((self.high*0.1)+(self.vwap*0.9))), rank(self.volume), 11)))*-1)
        return tech101.iloc[-self.length]

    # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    def tech102(self):
        tech102 = sma(np.maximum(self.volume-delay(self.volume, 1), 0),
                      6, 1)/sma(abs(self.volume-delay(self.volume, 1)), 6, 1)*100
        return tech102.iloc[-self.length]

    # ((20-LOWDAY(LOW,20))/20)*100
    def tech103(self):
        tech103 = ((20-lowday(self.low, 20))/20)*100
        return tech103.iloc[-self.length]

    # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
    def tech104(self):
        tech104 = (-1*(delta(correlation(self.high, self.volume, 5), 5)
                       * rank(stddev(self.close, 20))))
        return tech104.iloc[-self.length]

    # (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
    def tech105(self):
        tech105 = (-1*correlation(rank(self.open), rank(self.volume), 10))
        return tech105.iloc[-self.length]

    # CLOSE-DELAY(CLOSE,20)
    def tech106(self):
        tech106 = self.close-delay(self.close, 20)
        return tech106.iloc[-self.length]

    # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
    def tech107(self):
        tech107 = (((-1*rank((self.open-delay(self.high, 1))))*rank((self.open -
                     delay(self.close, 1))))*rank((self.open-delay(self.low, 1))))
        return tech107.iloc[-self.length]

    # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1)
    def tech108(self):
        tech108 = (pow(rank((self.high-ts_min(self.high, 2))),
                       rank(correlation((self.vwap), (mean(self.volume, 120)), 6)))*-1)
        return tech108.iloc[-self.length]

    # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
    def tech109(self):
        tech109 = sma(self.high-self.low, 10, 2) / \
            sma(sma(self.high-self.low, 10, 2), 10, 2)
        return tech109.iloc[-self.length]

    # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    def tech110(self):
        tech110 = ts_sum(np.maximum(0, self.high-delay(self.close, 1)), 20) / \
            ts_sum(np.maximum(0, delay(self.close, 1)-self.low), 20)*100
        return tech110.iloc[-self.length]

    # SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-L OW),4,2)
    def tech111(self):
        tech111 = sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low), 11, 2) - \
            sma(self.volume*((self.close-self.low) -
                             (self.high-self.close))/(self.high-self.low), 4, 2)
        return tech111.iloc[-self.length]

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
        return tech112.iloc[-self.length]

    # (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))
    def tech113(self):
        tech113 = (-1 * ((rank((ts_sum(delay(self.close, 5), 20) / 20)) * correlation(self.close,
                          self.volume, 2)) * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))))
        return tech113.iloc[-self.length]

    # ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    def tech114(self):
        tech114 = ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))
                    ) / (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))
        return tech114.iloc[-self.length]

    # RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7))
    def tech115(self):
        tech115 = pow(rank(correlation(((self.high*0.9)+(self.close*0.1)), mean(self.volume, 30), 10)),
                      rank(correlation(ts_rank(((self.high+self.low)/2), 4), ts_rank(self.volume, 10), 7)))
        return tech115.iloc[-self.length]

    # REGBETA(CLOSE,SEQUENCE,20) #
    def tech116(self):
        tech116 = regbeta(self.close, sequence(20), 20)
        return tech116.iloc[-self.length]

   # ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
    def tech117(self):
        tech117 = ((ts_rank(self.volume, 32) * (1 - ts_rank(((self.close +
                    self.high) - self.low), 16))) * (1 - ts_rank(self.returns, 32)))
        return tech117.iloc[-self.length]

    # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    def tech118(self):
        tech118 = ts_sum(self.high-self.open, 20) / \
            ts_sum(self.open-self.low, 20)*100
        return tech118.iloc[-self.length]

    # (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) -RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
    def tech119(self):
        tech119 = (rank(decay_linear(correlation(self.vwap, ts_sum(mean(self.volume, 5), 26), 5), 7)) - rank(
            decay_linear(ts_rank(ts_min(correlation(rank(self.open), rank(mean(self.volume, 15)), 21), 9), 7), 8)))
        return tech119.iloc[-self.length]

    # (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    def tech120(self):
        tech120 = (rank((self.vwap - self.close)) /
                   rank((self.vwap + self.close)))
        return tech120.iloc[-self.length]

    # ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)
    def tech121(self):
        tech121 = (pow(rank((self.vwap - ts_min(self.vwap, 12))), ts_rank(correlation(
            ts_rank(self.vwap, 20), ts_rank(mean(self.volume, 60), 2), 18), 3)) * -1)
        return tech121.iloc[-self.length]

    # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SM A(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    def tech122(self):
        tech122 = (sma(sma(sma(log(self.close), 13, 2), 13, 2), 13, 2)-delay(sma(sma(sma(log(self.close),
                   13, 2), 13, 2), 13, 2), 1))/delay(sma(sma(sma(log(self.close), 13, 2), 13, 2), 13, 2), 1)
        return tech122.iloc[-self.length]

    # ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
    def tech123(self):
        tech123 = ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(
            mean(self.volume, 60), 20), 9)) < rank(correlation(self.low, self.volume, 6))) * -1)
        return tech123.iloc[-self.length]

    # (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
    def tech124(self):
        tech124 = (self.close - self.vwap) / \
            decay_linear(rank(ts_max(self.close, 30)), 2)
        return tech124.iloc[-self.length]

    # (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))
    def tech125(self):
        tech125 = (rank(decay_linear(correlation((self.vwap), mean(self.volume, 80), 17), 20)
                        ) / rank(decay_linear(delta(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)))
        return tech125.iloc[-self.length]

    # (CLOSE + HIGH + LOW) / 3
    def tech126(self):
        tech126 = (self.close+self.high+self.low)/3
        return tech126.iloc[-self.length]

    # (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
    def tech127(self):
        tech127 = pow(mean(pow(
            100*(self.close-ts_max(self.close, 12))/(ts_max(self.close, 12)), 2), 12), 0.5)
        return tech127.iloc[-self.length]

    # 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/
    # SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0), 14)))
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
        return tech128.iloc[-self.length]

    # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    def tech129(self):
        cond1 = ((self.close-delay(self.close, 1)) < 0)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = abs(self.close-delay(self.close, 1))
        tech129 = ts_sum(part1, 12)
        return tech129.iloc[-self.length]

    # (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
    def tech130(self):
        tech130 = (rank(decay_linear(correlation(((self.high + self.low) / 2), mean(self.volume, 40), 9),
                                     10)) / rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 7), 3)))
        return tech130.iloc[-self.length]

    # (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
    def tech131(self):
        tech131 = pow(rank(delta(self.vwap, 1)), ts_rank(
            correlation(self.close, mean(self.volume, 50), 18), 18))
        return tech131.iloc[-self.length]

    # MEAN(AMOUNT, 20)
    def tech132(self):
        tech132 = mean(self.amount, 20)
        return tech132.iloc[-self.length]

    # ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    def tech133(self):
        tech133 = ((20-highday(self.high, 20))/20) * \
            100-((20-lowday(self.low, 20))/20)*100
        return tech133.iloc[-self.length]

    # (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    def tech134(self):
        tech134 = (self.close-delay(self.close, 12)) / \
            delay(self.close, 12)*self.volume
        return tech134.iloc[-self.length]

    # SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    def tech135(self):
        tech135 = sma(delay(self.close/delay(self.close, 20), 1), 20, 1)
        return tech135.iloc[-self.length]

    # ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    def tech136(self):
        tech136 = ((-1 * rank(delta(self.returns, 3))) *
                   correlation(self.open, self.volume, 10))
        return tech136.iloc[-self.length]

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
        return tech137.iloc[-self.length]

    # ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
    def tech138(self):
        tech138 = ((rank(decay_linear(delta((((self.low * 0.7) + (self.vwap * 0.3))), 3), 20)) - ts_rank(decay_linear(
            ts_rank(correlation(ts_rank(self.low, 8), ts_rank(mean(self.volume, 60), 17), 5), 19), 16), 7)) * -1)
        return tech138.iloc[-self.length]

    # (-1 * CORR(OPEN, VOLUME, 10))
    def tech139(self):
        tech139 = (-1 * correlation(self.open, self.volume, 10))
        return tech139.iloc[-self.length]

    # MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
    def tech140(self):
        tech140 = np.minimum(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8)),
                             ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(mean(self.volume, 60), 20), 8), 7), 3))
        return tech140.iloc[-self.length]

    # (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
    def tech141(self):
        tech141 = (
            rank(correlation(rank(self.high), rank(mean(self.volume, 15)), 9)) * -1)
        return tech141.iloc[-self.length]

    # (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
    def tech142(self):
        tech142 = (((-1 * rank(ts_rank(self.close, 10))) * rank(delta(delta(self.close, 1), 1)))
                   * rank(ts_rank((self.volume/mean(self.volume, 20)), 5)))
        return tech142.iloc[-self.length]

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
        return tech144.iloc[-self.length]

    # (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
    def tech145(self):
        tech145 = (mean(self.volume, 9) - mean(self.volume, 26)) / \
            mean(self.volume, 12) * 100
        return tech145.iloc[-self.length]

    # MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*(( CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOS E-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE, 1))/DELAY(CLOSE,1),61,2)))^2,60)
    def tech146(self):
        tech146 = (mean((self.close-delay(self.close, 1))/delay(self.close, 1)-sma((self.close-delay(self.close, 1))/delay(self.close, 1), 61, 2), 20) *
                   ((self.close-delay(self.close, 1))/delay(self.close, 1)-sma((self.close-delay(self.close, 1))/delay(self.close, 1), 61, 2)) /
                   sma(pow(((self.close-delay(self.close, 1))/delay(self.close, 1)-((self.close-delay(self.close, 1))/delay(self.close, 1)-sma((self.close-delay(self.close, 1))/delay(self.close, 1), 61, 2))), 2), 61, 2))
        return tech146.iloc[-self.length]

    # REGBETA(MEAN(CLOSE, 12), SEQUENCE(12))
    def tech147(self):
        tech147 = regbeta(mean(self.close, 12), sequence(12), 12)
        return tech147.iloc[-self.length]

    # ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    def tech148(self):
        tech148 = ((rank(correlation((self.open), ts_sum(mean(self.volume, 60), 9), 6)) < rank(
            (self.open - ts_min(self.open, 14)))) * -1)
        return tech148.iloc[-self.length]

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
        return tech150.iloc[-self.length]

    # SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)
    def tech151(self):
        tech151 = sma(self.close - delay(self.close, 20), 20, 1)
        return tech151.iloc[-self.length]

    # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
    def tech152(self):
        tech152 = sma(mean(delay(sma(delay(self.close/delay(self.close, 9), 1), 9, 1), 1), 12) -
                      mean(delay(sma(delay(self.close/delay(self.close, 9), 1), 9, 1), 1), 26), 9, 1)
        return tech152.iloc[-self.length]

    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    def tech153(self):
        tech153 = (mean(self.close, 3)+mean(self.close, 6) +
                   mean(self.close, 12)+mean(self.close, 24))/4
        return tech153.iloc[-self.length]

    # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))
    def tech154(self):
        tech154 = (((self.vwap-ts_min(self.vwap, 16)) <
                    (correlation(self.vwap, mean(self.vwap, 180), 18)))*1)
        return tech154.iloc[-self.length]

    # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    def tech155(self):
        tech155 = sma(self.volume, 13, 2)-sma(self.volume, 27, 2) - \
            sma(sma(self.volume, 13, 2)-sma(self.volume, 27, 2), 10, 2)
        return tech155.iloc[-self.length]

    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1
    def tech156(self):
        tech156 = (np.maximum(rank(decay_linear(delta(self.vwap, 5), 3)), rank(decay_linear(
            ((delta(((self.open*0.15)+(self.low*0.85)), 2)/((self.open*0.15)+(self.low*0.85)))*-1), 3)))*-1)
        return tech156.iloc[-self.length]

    # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5))
    def tech157(self):
        tech157 = (ts_min(product(rank(rank(log(ts_sum(ts_min(rank(rank(
            (-1*rank(delta((self.close-1), 5))))), 2), 1)))), 1), 5)+ts_rank(delay((-1*self.returns), 6), 5))
        return tech157.iloc[-self.length]

    # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    def tech158(self):
        tech158 = ((self.high-sma(self.close, 15, 2)) -
                   (self.low-sma(self.close, 15, 2)))/self.close
        return tech158.iloc[-self.length]

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
        return tech159.iloc[-self.length]

    # SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def tech160(self):
        cond1 = self.close <= delay(self.close, 1)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = stddev(self.close, 20)
        tech160 = sma(part1, 20, 1)
        return tech160.iloc[-self.length]

    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    def tech161(self):
        tech161 = mean(np.maximum(np.maximum((self.high-self.low), abs(
            delay(self.close, 1)-self.high)), abs(delay(self.close, 1)-self.low)), 12)
        return tech161.iloc[-self.length]

    # (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOS E-DELAY(CLOSE,1),0),12,1)/
    # SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(C LOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-
    # MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    def tech162(self):
        tech162 = ((sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100 -
                    ts_min(sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100, 12)) /
                   (ts_max(sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100, 12) -
                    ts_min(sma(np.maximum(self.close-delay(self.close, 1), 0), 12, 1)/sma(delay(self.close-delay(self.close, 1)), 12, 1)*100, 12)))
        return tech162.iloc[-self.length]

    # RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    def tech163(self):
        tech163 = rank(((((-1 * self.returns) * mean(self.volume, 20))
                         * self.vwap) * (self.high - self.close)))
        return tech163.iloc[-self.length]

    # SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-D ELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
    def tech164(self):
        cond1 = self.close > delay(self.close, 1)
        part1 = pd.DataFrame(np.ones(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1/(self.close-delay(self.close, 1))
        tech164 = sma((part1-ts_min(part1, 12)) /
                      (self.high-self.low)*100, 13, 2)
        return tech164.iloc[-self.length]

    # # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    # def tech165(self):
    #     # tech165 = np.maximum(np.cumsum(self.close-mean(self.close,48)),0)-np.minimum(np.cumsum(self.close-mean(self.close,48)),0)/stddev(self.close,48)
    #     return 0

    # -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    def tech166(self):
        tech166 = -20*pow(20-1, 1.5)*ts_sum(self.close/delay(self.close, 1)-1-mean(self.close/delay(self.close, 1)-1,
                                                                                   20), 20)/((20-1)*(20-2)*pow(ts_sum(pow(self.close/delay(self.close, 1), 20, 2), 20), 1.5))
        return tech166.iloc[-self.length]

    # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    def tech167(self):
        cond1 = (self.close-delay(self.close, 1)) > 0
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.close-delay(self.close, 1)
        tech167 = ts_sum(part1, 12)
        return tech167.iloc[-self.length]

    # (-1*VOLUME/MEAN(VOLUME,20))
    def tech168(self):
        tech168 = (-1*self.volume/mean(self.volume, 20))
        return tech168.iloc[-self.length]

    # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)
    def tech169(self):
        tech169 = sma(mean(delay(sma(self.close-delay(self.close, 1), 9, 1), 1), 12) -
                      mean(delay(sma(self.close-delay(self.close, 1), 9, 1), 1), 26), 10, 1)
        return tech169.iloc[-self.length]

    # ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))
    def tech170(self):
        tech170 = ((((rank((1 / self.close)) * self.volume) / mean(self.volume, 20)) * ((self.high * rank(
            (self.high - self.close))) / (ts_sum(self.high, 5) / 5))) - rank((self.vwap - delay(self.vwap, 5))))
        return tech170.iloc[-self.length]

    # ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    def tech171(self):
        tech171 = ((-1 * ((self.low - self.close) * pow(self.open, 5))
                    ) / ((self.close - self.high) * pow(self.close, 5)))
        return tech171.iloc[-self.length]

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
        return tech172.iloc[-self.length]

    # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);
    def tech173(self):
        tech173 = 3*sma(self.close, 13, 2)-2*sma(sma(self.close, 13, 2),
                                                 13, 2)+sma(sma(sma(log(self.close), 13, 2), 13, 2), 13, 2)
        return tech173.iloc[-self.length]

    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def tech174(self):
        cond1 = self.close > delay(self.close, 1)
        part1 = pd.DataFrame(np.zeros(self.close.shape),
                             index=self.close.index, columns=self.close.columns)
        part1[cond1] = stddev(self.close, 20)
        tech174 = sma(part1, 20, 1)
        return tech174.iloc[-self.length]

    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    def tech175(self):
        tech175 = mean(np.maximum(np.maximum((self.high-self.low), abs(
            delay(self.close, 1)-self.high)), abs(delay(self.close, 1)-self.low)), 6)
        return tech175.iloc[-self.length]

    # CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    def tech176(self):
        tech176 = correlation(rank(((self.close - ts_min(self.low, 12)) / (
            ts_max(self.high, 12) - ts_min(self.low, 12)))), rank(self.volume), 6)
        return tech176.iloc[-self.length]

    # ((20-HIGHDAY(HIGH,20))/20)*100
    def tech177(self):
        tech177 = ((20-highday(self.high, 20))/20)*100
        return tech177.iloc[-self.length]

    # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    def tech178(self):
        tech178 = (self.close-delay(self.close, 1)) / \
            delay(self.close, 1)*self.volume
        return tech178.iloc[-self.length]

    # (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    def tech179(self):
        tech179 = (rank(correlation(self.vwap, self.volume, 4)) *
                   rank(correlation(rank(self.low), rank(mean(self.volume, 50)), 12)))
        return tech179.iloc[-self.length]

    # ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))
    def tech180(self):
        cond1 = mean(self.volume, 20) < self.volume
        tech180 = (-1*self.volume)
        tech180[cond1] = (
            (-1 * ts_rank(abs(delta(self.close, 7)), 60)) * sign(delta(self.close, 7)))
        return tech180.iloc[-self.length]

    # SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/
    # SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
    def tech181(self):
        part1 = pd.DataFrame(np.tile(pow(self.benchmark_close-mean(self.benchmark_close, 20), 2),
                                     (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T
        part2 = pd.DataFrame(np.tile(ts_sum(pow(self.benchmark_close-mean(self.benchmark_close, 20), 3), 20),
                                     (self.close.shape[1], 1)), index=self.close.columns, columns=self.close.index).T
        tech181 = ts_sum(((self.close/delay(self.close, 1)-1) -
                          mean((self.close/delay(self.close, 1)-1), 20))-part1, 20)/part2
        return tech181.iloc[-self.length]

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
        return tech182.iloc[-self.length]

    # # MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    # def tech183(self):
    #     # tech183 = np.maximum(np.cumsum(self.close-mean(self.close,24)),0)-np.minimum(np.cumsum(self.close-mean(self.close,24)),0)/stddev(self.close,24)
    #     # save_hdf(tech183, 'tech183')
    #     return 0

    # (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    def tech184(self):
        tech184 = (rank(correlation(delay((self.open - self.close), 1),
                                    self.close, 200)) + rank((self.open - self.close)))
        return tech184.iloc[-self.length]

    # RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
    def tech185(self):
        tech185 = rank((-1 * (pow(1 - (self.open / self.close), 2))))
        return tech185.iloc[-self.length]

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
        return tech186.iloc[-self.length]

    # SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    def tech187(self):
        cond1 = self.open <= delay(self.open, 1)
        tech187 = np.maximum((self.high-self.open),
                             (self.open-delay(self.open, 1)))
        tech187[cond1] = 0
        return tech187.iloc[-self.length]

    # ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    def tech188(self):
        tech188 = ((self.high-self.low-sma(self.high-self.low, 11, 2)
                    )/sma(self.high-self.low, 11, 2))*100
        return tech188.iloc[-self.length]

    # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    def tech189(self):
        tech189 = mean(abs(self.close-mean(self.close, 6)), 6)
        return tech189.iloc[-self.length]

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
        return tech190.iloc[-self.length]

    # (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE
    def tech191(self):
        tech191 = (correlation(mean(self.volume, 20), self.low, 5) +
                   ((self.high + self.low) / 2)) - self.close
        return tech191.iloc[-self.length]

    # -1*CORR(VWAP,VOLUME,6)
    def techJLBL(self):
        techJLBL = -1 * correlation(self.vwap, self.volume, 6)
        return techJLBL.iloc[-self.length]

    # OPEN/CLOSE
    def techKPQK(self):
        techKPQK = self.open/delay(self.close, 1)
        return techKPQK.iloc[-self.length]

    # -1*VOLUME/MEAN(VOLUME,20)
    def techYCCJL(self):
        techYCCJL = -1*self.volume/mean(self.volume, 6)
        return techYCCJL.iloc[-self.length]

    # -1*CORR(HIGH/LOW,VOLUME,6)
    def techLFBL(self):
        techLFBL = -1 * correlation(self.high/self.low, self.volume, 6)
        return techLFBL.iloc[-self.length]

    '''==================================================================================
                                  Barra9个风格因子从这里开始
        =================================================================================='''
    def beta(self, window=250):
        w = (0.5 ** (np.array(list(range(1, 251, 1))) / 60))[::-1]
        rows = self.returns.shape[0]
        columns = self.returns.columns
        beta = np.ones(self.returns.shape) * np.nan
        for i, col in enumerate(columns):
            # print(col)
            for j in range(window - 1, rows):
                model = OLS(x=np.array(self.benchmark_ret[j - window + 1:j + 1]) * w,
                            y=np.array(self.returns[col][j - window + 1:j + 1]) * w)
                beta[j, i] = model.beta
                self.e[j, i] = model.std_err
        beta = pd.DataFrame(beta, index=self.returns.index, columns=self.returns.columns)
        beta = beta.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        beta = remove_extreme(beta, axis=1)  # 三倍标准差去极值
        beta = pd.DataFrame(preprocessing.scale(beta, axis=1), index=self.returns.index,
                            columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
        self.e = pd.DataFrame(self.e, index=self.returns.index, columns=self.returns.columns)
        return beta.iloc[-self.length]

    def momentum(self, window=520):
        w = np.reshape((0.5 ** (np.array(list(range(1, 501, 1))) / 120))[::-1], (500, 1))
        rstr = np.ones(self.returns.shape) * np.nan
        rows = self.returns.shape[0]
        for i in range(window - 1, rows):
            rstr[i,] = np.sum(np.log(np.array(self.returns.iloc[i - window + 1:i - 20 + 1]) + 1) * w, axis=0)
        momentum = pd.DataFrame(rstr, index=self.returns.index, columns=self.returns.columns)
        momentum = momentum.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)
        momentum = remove_extreme(momentum, axis=1)  # 三倍标准差去极值
        momentum = pd.DataFrame(preprocessing.scale(momentum, axis=1), index=self.returns.index,
                                columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
        return momentum.iloc[-self.length]

    def size(self):
        lncap = np.log(self.total_mv)
        size = pd.DataFrame(lncap, index=self.returns.index, columns=self.returns.columns)  # .iloc[-self.length:,:]
        size = size.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        size = remove_extreme(size, axis=1)  # 三倍标准差去极值
        size = pd.DataFrame(preprocessing.scale(size, axis=1), index=self.returns.index,
                            columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
        return size.iloc[-self.length]

    def earningsyield(self):
        epibs = np.divide(self.eps_FY1, self.close)
        epibs = epibs.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        epibs = remove_extreme(epibs, axis=1)  # 三倍标准差去极值
        epibs = pd.DataFrame(preprocessing.scale(epibs, axis=1), index=self.returns.index,
                             columns=self.returns.columns)
        # ==============================================================
        etop = np.divide(1, self.pe_ttm)
        etop = etop.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        etop = remove_extreme(etop, axis=1)  # 三倍标准差去极值
        etop = pd.DataFrame(preprocessing.scale(etop, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ==============================================================
        cetop = np.divide(self.cfps, self.close)
        cetop = cetop.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        cetop = remove_extreme(cetop, axis=1)  # 三倍标准差去极值
        cetop = pd.DataFrame(preprocessing.scale(cetop, axis=1), index=self.returns.index,
                             columns=self.returns.columns)
        # ==============================================================
        earningsyield = 0.68 * epibs + 0.11 * etop + 0.21 * cetop
        earningsyield = earningsyield.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        earningsyield = remove_extreme(earningsyield, axis=1)  # 三倍标准差去极值
        earningsyield = pd.DataFrame(preprocessing.scale(earningsyield, axis=1), index=self.returns.index,
                                     columns=self.returns.columns)
        return earningsyield.iloc[-self.length]

    # 波动率因子
    def volatility(self, window=250):
        w = np.reshape((0.5 ** (np.array(list(range(1, 251, 1))) / 40))[::-1], (250, 1))
        dastd = np.ones(self.returns.shape) * np.nan
        rows = self.returns.shape[0]
        for i in range(window - 1, rows):
            dastd[i,] = pow(np.sum(pow((np.array(self.returns.iloc[i - window + 1:i + 1]) - np.array(
                self.returns.iloc[i - window + 1:i + 1]).mean(axis=0)), 2) * w, axis=0), 1 / 2)
        dastd = pd.DataFrame(dastd, index=self.returns.index, columns=self.returns.columns)  # .iloc[-self.length:,:]
        dastd = dastd.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        dastd = remove_extreme(dastd, axis=1)  # 三倍标准差去极值
        dastd = pd.DataFrame(preprocessing.scale(dastd, axis=1), index=self.returns.index,
                             columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
        # =============================================================
        cmra = np.ones(self.returns.shape) * np.nan
        for i in range(window + 2, rows):
            temp = []  # 存储12个月收益率,每月以21天计算
            for m in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
                month_return = []  # 单月收益率
                a = self.returns.iloc[i - 21 * m:i - 21 * (m - 1), :]
                month_return = np.log(np.sum(self.returns.iloc[i - 21 * m:i - 21 * (m - 1), :], axis=0) + 1)
                temp.append(month_return)
            temp = np.cumsum(np.array(temp), axis=0)
            cmra[i,] = np.log(1 + np.max(temp, axis=0)) - np.log(1 + np.min(temp, axis=0))
        cmra = pd.DataFrame(cmra, index=self.returns.index, columns=self.returns.columns)
        cmra = cmra.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)
        cmra = remove_extreme(cmra, axis=1)
        cmra = pd.DataFrame(preprocessing.scale(cmra, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ==============================================================
        hsigma = self.e
        hsigma = hsigma.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)
        hsigma = remove_extreme(hsigma, axis=1)
        hsigma = pd.DataFrame(preprocessing.scale(hsigma, axis=1), index=self.returns.index,
                         columns=self.returns.columns)
        # ==============================================================
        volatility = 0.74 * dastd + 0.16 * cmra + 0.1 * hsigma
        volatility = volatility.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        volatility = remove_extreme(volatility, axis=1)  # 三倍标准差去极值
        volatility = pd.DataFrame(preprocessing.scale(volatility, axis=1), index=self.returns.index,
                                  columns=self.returns.columns)
        return volatility.iloc[-self.length]

    def growth(self):
        sgro = self.cagrgr_PY5
        sgro = sgro.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        sgro = remove_extreme(sgro, axis=1)  # 三倍标准差去极值
        sgro = pd.DataFrame(preprocessing.scale(sgro, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # =============================================================
        egro = self.cagrpni_PY5
        egro = egro.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        egro = remove_extreme(egro, axis=1)  # 三倍标准差去极值
        egro = pd.DataFrame(preprocessing.scale(egro, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # =============================================================
        egib = self.sestni_YOY3
        egib = egib.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        egib = remove_extreme(egib, axis=1)  # 三倍标准差去极值
        egib = pd.DataFrame(preprocessing.scale(egib, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # =============================================================
        egib_s = self.sestni_YOY1
        egib_s = egib_s.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        egib_s = remove_extreme(egib_s, axis=1)  # 三倍标准差去极值
        egib_s = pd.DataFrame(preprocessing.scale(egib_s, axis=1), index=self.returns.index,
                              columns=self.returns.columns)
        # =============================================================
        growth = 0.47 * sgro + 0.24 * egro + 0.18 * egib + 0.11 * egib_s
        growth = growth.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        growth = remove_extreme(growth, axis=1)  # 三倍标准差去极值
        growth = pd.DataFrame(preprocessing.scale(growth, axis=1), index=self.returns.index,
                              columns=self.returns.columns)
        return growth.iloc[-self.length]

    def value(self):
        btop = np.divide(self.total_equity, self.total_mv)
        value = btop.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        value = remove_extreme(value, axis=1)  # 三倍标准差去极值
        value = pd.DataFrame(preprocessing.scale(value, axis=1), index=self.returns.index,
                             columns=self.returns.columns)
        return value.iloc[-self.length]

    def leverage(self):
        mlev = np.divide((self.total_mv + self.total_ncl), self.total_mv)
        mlev = mlev.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        mlev = remove_extreme(mlev, axis=1)  # 三倍标准差去极值
        mlev = pd.DataFrame(preprocessing.scale(mlev, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ===============================================================
        dtoa = np.divide(self.total_liab, self.total_assets)
        dtoa = dtoa.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        dtoa = remove_extreme(dtoa, axis=1)  # 三倍标准差去极值
        dtoa = pd.DataFrame(preprocessing.scale(dtoa, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ===============================================================
        blev = np.divide((self.total_equity + self.total_ncl), self.total_equity)
        blev = blev.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        blev = remove_extreme(blev, axis=1)  # 三倍标准差去极值
        blev = pd.DataFrame(preprocessing.scale(blev, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ===============================================================
        leverage = 0.38 * mlev + 0.35 * dtoa + 0.27 * blev
        leverage = leverage.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        leverage = remove_extreme(leverage, axis=1)  # 三倍标准差去极值
        leverage = pd.DataFrame(preprocessing.scale(leverage, axis=1), index=self.returns.index,
                                columns=self.returns.columns)
        return leverage.iloc[-self.length]

    def liquidity(self):
        stom = np.log(ts_sum(self.turnover_rate, 21))
        stom = stom.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        stom = remove_extreme(stom, axis=1)  # 三倍标准差去极值
        stom = pd.DataFrame(preprocessing.scale(stom, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ===============================================================
        stoq = np.log((1 / 3) * (ts_sum(self.turnover_rate, 63)))
        stoq = stoq.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        stoq = remove_extreme(stoq, axis=1)  # 三倍标准差去极值
        stoq = pd.DataFrame(preprocessing.scale(stoq, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ===============================================================
        stoa = np.log((1 / 12) * (ts_sum(self.turnover_rate, 252)))
        stoa = stoa.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        stoa = remove_extreme(stoa, axis=1)  # 三倍标准差去极值
        stoa = pd.DataFrame(preprocessing.scale(stoa, axis=1), index=self.returns.index,
                            columns=self.returns.columns)
        # ===============================================================
        liquidity = 0.35 * stom + 0.35 * stoq + 0.3 * stoa
        liquidity = liquidity.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
        liquidity = remove_extreme(liquidity, axis=1)  # 三倍标准差去极值
        liquidity = pd.DataFrame(preprocessing.scale(liquidity, axis=1), index=self.returns.index,
                                 columns=self.returns.columns)
        return liquidity.iloc[-self.length]

    def methods(self):
        return (list(filter(lambda m: not m.startswith("__") and not m.endswith("__") and callable(getattr(self, m)), dir(self))))


def run_func(paras):
    startdate = paras["startdate"]
    enddate = paras["enddate"]
    count = paras["count"]
    length = paras["length"]
    code = paras["code"]
    alphas = AlphaFactor(code, startdate,enddate, count, length)
    func_list = [x for x in alphas.methods() if x not in ["methods"]]
    factor_data = {}  # 存放当日因子存放数据
    for func_name in func_list:
        factor_data[func_name] = eval("alphas." + func_name + "()")
    save_h5(factor_data,date)
    return

def set_params(TradingDay):
    td = {"startdate":None,
          "count":550,
          "length":1,}
    params = []
    for i, date in enumerate(TradingDay):
        td["enddate"] = date[0]
        td['code'] = get_stock(date[0])
        params.append(td.copy())
    return params


if __name__ == '__main__':
    """设置更新日期"""
    pro = ts.pro_api()
    """对每个交易日进行单独计算"""
    TradingDay = [str(x) for x in pd.read_csv(Pre_path + "\\TradeDate.csv")["trade_date"].tolist()]
    for date in TradingDay:
        Code = get_stock(date="20191212", ST=False, Suspend=False) # 获取股票池
        Start_Date = None      # 起始日期
        End_Date = date        # 截止日期
        count = 550  # 用于计算因子的数据长度
        length = 1   # 输出数据长度，每日保存
        alphas = AlphaFactor(Code, Start_Date, End_Date , count, length)
        func_list =[x for x in alphas.methods() if x not in ["methods"]]
        # 单进程方案，效率低无法利用多核
        factor_data = {} # 存放当日因子存放数据
        for func_name in func_list:
            factor_data[func_name] = eval("alphas." + func_name + "()")
        save_h5(factor_data,date)
    """多进程处理方案，如果不采用共享内存方案则吃不消"""
    # TradingDay = pro.trade_cal(exchange='SSE', start_date="20100101", end_date="20191217", is_open=1).cal_date.tolist()
    # TradingDay = [TradingDay[i:i + 1] for i in range(0, len(TradingDay), 1)]
    # paras = set_params(TradingDay)
    # pool = multiprocessing.Pool(5)
    # pool.map(run_func, paras)
    # pool.close()
    # pool.join()
    # print("The data of Alpha101 update completely!")
