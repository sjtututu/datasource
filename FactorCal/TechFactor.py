#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/5 15:12
# @Author: Tu Xinglong
# @File  : TechFactor.py

import os
import sys

import pandas as pd
import numpy as np
import itertools
from numpy import abs
from numpy import log
from numpy import sign
import multiprocessing
import tushare as ts

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa


class TechFactor(object):
    '''
    传统技术面因子
    '''
    def __init__(self, startdate, enddate, count, length):
        '''
        获取数据信息
        :param self_data:
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
        self.vwap = stock_price['amount']/ (stock_price['volume'] + 0.001)
        self.length = length
        np.seterr(divide='ignore', invalid='ignore')   # 忽略警告

    def tech_acd(self,n1=6,n2=20):
        '''
        LC=REF(CLOSE,1)
        DIF=CLOSE-IF(CLOSE>LC,MIN(LOW,LC),MAX(HIGH,LC))
        ACD=SUM(IF(CLOSE=LC,0,DIF),0)
        :return:
        '''
        cond1 = self.close > delay(self.close,1)
        part1 = np.maximum(self.high,delay(self.close,1))
        part1[cond1] =np.minimum(self.low,delay(self.close,1))
        dif = self.close - part1
        cond2 = (self.close == delay(self.close,1))
        part2 = dif
        part2[cond2] = 0
        tech_acd6 = ts_sum(part2,n1)
        save_hdf(tech_acd6, 'tech_acd6', self.length)
        tech_acd20 = ts_sum(part2, n2)
        save_hdf(tech_acd20, 'tech_acd20', self.length)
        return


    def tech_adtm(self, n=23, m=8):
        """
        动态买卖气指标	adtm(23,8)
        如果开盘价≤昨日开盘价，DTM=0
        如果开盘价＞昨日开盘价，DTM=(最高价-开盘价)和(开盘价-昨日开盘价)的较大值
        如果开盘价≥昨日开盘价，DBM=0
        如果开盘价＜昨日开盘价，DBM=(开盘价-最低价)
        STM=DTM在N日内的和
        SBM=DBM在N日内的和
        如果STM > SBM,ADTM=(STM-SBM)/STM
        如果STM < SBM , ADTM = (STM-SBM)/SBM
        如果STM = SBM,ADTM=0
        ADTMMA=MA(ADTM,M)
        """
        cond1 = (self.open <= delay(self.open, 1))
        DTM = np.maximum((self.high - self.open), (self.open - delay(self.open, 1)))
        DTM[cond1] = 0
        save_hdf(DTM, 'tech_dtm', self.length)
        cond2 = (self.open >= delay(self.open, 1))
        DBM = (self.open - self.low)
        DBM[cond2] = 0
        save_hdf(DBM, 'tech_dbm', self.length)
        cond3 = ts_sum(DTM, n) > ts_sum(DBM, n)
        cond4 = ts_sum(DTM, n) < ts_sum(DBM, n)
        tech_adtm = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        tech_adtm[cond3] = (ts_sum(DTM, n) - ts_sum(DBM, n)) / ts_sum(DTM, n)
        tech_adtm[cond4] = (ts_sum(DTM, n) - ts_sum(DBM, n)) / ts_sum(DBM, n)
        tech_adtmma = ma(tech_adtm,m)
        save_hdf(tech_adtm, 'tech_adtm', self.length)
        save_hdf(tech_adtmma, 'tech_adtmma', self.length)
        return


    def tech_ad(self,n1=6,n2=20):
        """
        AD指标将每日的成交量通过价格加权累计，用以计算成交量的动量。
        AD指标 = SUM((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOL, 0)
        为了使指标具有可比性， 我们分别计算了6个和一个交易月的AD指标累加值。
        """
        tech_ad=((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume
        save_hdf(tech_ad, 'tech_ad', self.length)
        tech_adsum6=ts_sum(tech_ad,6)
        save_hdf(tech_adsum6, 'tech_adsum6', self.length)
        tech_adsum20=ts_sum(tech_ad,20)
        save_hdf(tech_adsum20, 'tech_adsum20', self.length)
        return

    def tech_arbr(self, n=26):
        """
        人气意愿指标	arbr(26)
        N日AR=N日内（H－O）之和除以N日内（O－L）之和
        其中，H为当日最高价，L为当日最低价，O为当日开盘价，N为设定的时间参数，一般原始参数日设定为26日
        N日BR=N日内（H－CY）之和除以N日内（CY－L）之和
        其中，H为当日最高价，L为当日最低价，CY为前一交易日的收盘价，N为设定的时间参数，一般原始参数日设定为26日。
        """
        tech_ar = (self.high - self.open).rolling(n).sum() / (self.open - self.low).rolling(n).sum() * 100
        save_hdf(tech_ar, 'tech_ar', self.length)
        tech_br = (self.high - self.close.shift(1)).rolling(n).sum() / (self.close.shift() - self.low).rolling(n).sum() * 100
        save_hdf(tech_br, 'tech_br', self.length)
        return

    def tech_aroon(self,n=25):
        """
        Aroon上升数 = [(计算期天数 - 最高价后的天数) / 计算期天数] * 100
        Aroon下降数
        Aroon下降数 = [(计算期天数 - 最低价后的天数) / 计算期天数] * 100
        Aroon指标
        Aroon指标 = Aroon上升数 - Aroon下降数
        :return:
        """
        aroon_up = (n-highday(self.high,n))/n *100
        aroon_down =(n-lowday(self.high,n))/n *100
        tech_aroon = aroon_up - aroon_down
        save_hdf(tech_aroon, 'tech_aroon', self.length)
        return


    def tech_asi(self, n=5):
        """
        振动升降指标(累计震动升降因子) ASI  # 同花顺给出的公式不完整就不贴出来了
        LC=REF(C,1)
        AA=ABS(H-LC);BB=ABS(L-LC);CC=ABS(H-REF(L,1));DD=ABS(LC-REF(O,1))
        R=IF(AA>BB AND AA>CC,AA+BB/2+DD/4,IF(BB>CC AND
        BB>AA,BB+AA/2+DD/4,CC+DD/4))
        X=(C-LC+(C-O)/2+LC-REF(O,1))
        SI=16*X/R*MAX(AA,BB)
        ASI=SUM(SI,0)
        """
        aa = abs(self.high - delay(self.close, 1))
        bb = abs(self.low - delay(self.close, 1))
        cc = abs(self.high - delay(self.low, 1))
        dd = abs(delay(self.close, 1) - delay(self.open, 1))
        cond1 = (aa > bb) & (aa > cc)
        cond2 = (bb > cc) & (bb > aa)
        part1 = cc + dd / 4
        part1[cond1] = aa + bb / 2 + dd / 4
        part1[cond2] = bb + aa / 2 + dd / 4
        part2 = (self.close - delay(self.close, 1) +
                 (self.close - self.open) / 2 + delay(self.close, 1) -
                 delay(self.open, 1))
        si = 16 * part2 / part1 * np.maximum(aa, bb)
        tech_asi = ma(si, n)
        save_hdf(tech_asi, 'tech_asi', self.length)
        return

    def tech_atr(self, n=14):
        """
        真实波幅	atr(14)
        TR:MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
        ATR:MA(TR,N)
        """
        tr = np.maximum(np.maximum((self.high-self.low),abs(delay(self.close,1)-self.high)),abs(delay(self.close,1)-self.low))
        tech_atr = mean(tr,n)
        save_hdf(tech_atr, 'tech_atr', self.length)
        return

    def tech_bbi(self):
        """
        多空指数	BBI(3,6,12,24)
        BBI=（3日均价+6日均价+12日均价+24日均价）/4
        """
        tech_bbi= (ma(self.close, 3) + ma(self.close, 6) + ma(self.close, 12) + ma(self.close, 24)) / 4
        save_hdf(tech_bbi, 'tech_bbi', self.length)
        return


    def tech_bias(self, n1=6, n2=12, n3=24):
        """
        乖离率 bias
        bias=[(当日收盘价-12日平均价)/12日平均价]×100%
        """
        tech_bias6 = (np.true_divide((self.close -  mean(self.close,n1)),  mean(self.close,n1))) * 100
        save_hdf(tech_bias6, 'tech_bias6', self.length)
        tech_bias12 = (np.true_divide((self.close - mean(self.close, n2)), mean(self.close, n2))) * 100
        save_hdf(tech_bias12, 'tech_bias12', self.length)
        tech_bias24 = (np.true_divide((self.close - mean(self.close, n3)), mean(self.close, n3))) * 100
        save_hdf(tech_bias24, 'tech_bias24', self.length)
        return

    def tech_macd(self, n=12, m=26, k=9):
        """
        平滑异同移动平均线(Moving Average Convergence Divergence)
        今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）
        DIFF= EMA（N1）- EMA（N2）
        DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)
        MACD（BAR）=2×（DIF-DEA）
        return:
              osc: MACD bar / OSC 差值柱形图 DIFF - DEM
              diff: 差离值
              dea: 讯号线
        """
        diff = ema(self.close, n) - ema(self.close, m)
        dea = ema(diff, k)
        tech_macd =2 * (diff - dea)
        save_hdf(tech_macd, 'tech_macd', self.length)
        return

    def tech_kdj(self, n=9):
        """
        随机指标KDJ
        N日RSV=（第N日收盘价-N日内最低价）/（N日内最高价-N日内最低价）×100%
        当日K值=2/3前1日K值+1/3×当日RSV=SMA（RSV,M1）
        当日D值=2/3前1日D值+1/3×当日K= SMA（K,M2）
        当日J值=3 ×当日K值-2×当日D值
        """
        rsv = (self.close - ts_min(self.close,n)) / (ts_max(self.high,n) - ts_min(self.low,n)) * 100
        tech_kdj_k = sma(rsv, 3, 1)
        tech_kdj_d = sma(tech_kdj_k, 3, 1)
        tech_kdj_j = 3 * tech_kdj_k - 2 * tech_kdj_d
        save_hdf(tech_kdj_j, 'tech_kdj_j', self.length)
        return

    def tech_rsi(self, n=6):
        """
        相对强弱指标（Relative Strength Index，简称RSI
        LC= REF(CLOSE,1)
        RSI=SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N1,1)×100
        SMA（C,N,M）=M/N×今日收盘价+(N-M)/N×昨日SMA（N）
        """
        px = self.close - delay(self.close,1)
        px[px < 0] = 0
        tech_rsi = sma(px, n, 1) / sma(abs(self.close - delay(self.close,1)), n, 1) * 100
        save_hdf(tech_rsi, 'tech_rsi', self.length)
        return


    def tech_vrsi(self, n=6):
        """
        量相对强弱指标
        VRSI=SMA（最大值（成交量-REF（成交量，1），0），N,1）/SMA（ABS（（成交量-REF（成交量，1），N，1）×100%
        """
        px = self.volume - delay(self.volume,1)
        px[px < 0] = 0
        tech_vrsi = sma(px, n,1) / sma(abs(self.volume - delay(self.volume,1)), n, 1) * 100
        save_hdf(tech_vrsi, 'tech_vrsi', self.length)
        return


    def tech_boll(self, n=26, k=2):
        """
        布林线指标BOLL boll(26,2)	MID=MA(N)
        标准差MD=根号[∑（CLOSE-MA(CLOSE，N)）^2/N]
        UPPER=MID＋k×MD
        LOWER=MID－k×MD
        """
        tech_boll_up = ma(self.close, n) + k * md(self.close, n)
        tech_boll_low = ma(self.close, n) - k * md(self.close, n)
        save_hdf(tech_boll_up, 'tech_boll_up', self.length)
        save_hdf(tech_boll_low, 'tech_boll_low', self.length)
        return


    def tech_bbiboll(self, n=10, k=3):
        """
        BBI多空布林线	bbiboll(10,3)
        BBI={MA(3)+ MA(6)+ MA(12)+ MA(24)}/4
        标准差MD=根号[∑（BBI-MA(BBI，N)）^2/N]
        UPR= BBI＋k×MD
        DWN= BBI－k×MD
        """
        bbi = (ma(self.close, 3) + ma(self.close, 6) + ma(self.close, 12) + ma(self.close, 24)) / 4
        tech_bbiboll_upr = bbi + k * md(bbi, n)
        tech_bbiboll_dwn = bbi - k * md(bbi, n)
        save_hdf(tech_bbiboll_upr, 'tech_bbiboll_upr', self.length)
        save_hdf(tech_bbiboll_dwn, 'tech_bbiboll_dwn', self.length)
        return

    def tech_wr(self, n=14):
        """
        威廉指标 w&r
        WR=[最高值（最高价，N）-收盘价]/[最高值（最高价，N）-最低值（最低价，N）]×100%
        """
        tech_wr = (ts_max(self.high,n) - self.close) / (ts_max(self.high,n) - ts_min(self.low,n)) * 100
        save_hdf(tech_wr, 'tech_wr', self.length)
        return


    def tech_vr_rate(self, n=26):
        """
        成交量变异率 vr or vr_rate
        VR=（AVS+1/2CVS）/（BVS+1/2CVS）×100
        其中：
        AVS：表示N日内股价上涨成交量之和
        BVS：表示N日内股价下跌成交量之和
        CVS：表示N日内股价不涨不跌成交量之和
        """
        cond1 = self.close>delay(self.close,1)
        cond2 = self.close < delay(self.close, 1)
        cond3 = (self.close == delay(self.close, 1))
        part1 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part3 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1[cond1] = self.volume
        part2[cond2] = self.volume
        part3[cond3] = self.volume
        tech_vr_rate = (ts_sum(part1,n) + 1/2*ts_sum(part3,n))/(ts_sum(part2,n) + 1/2*ts_sum(part3,n)) *100
        save_hdf(tech_vr_rate, 'tech_vr_rate', self.length)
        return


    def tech_vr(self, n=5):
        """
        开市后平均每分钟的成交量与过去5个交易日平均每分钟成交量之比
        量比:=V/REF(MA(V,5),1);
        涨幅:=(C-REF(C,1))/REF(C,1)*100;
        1)量比大于1.8，涨幅小于2%，现价涨幅在0—2%之间，在盘中选股的
        选股:量比>1.8 AND 涨幅>0 AND 涨幅<2;
        """
        tech_vr = self.volume / ma(self.volume, n).shift(1)
        tech_rr= (self.close - self.close.shift(1)) / self.close.shift(1) * 100
        save_hdf(tech_vr, 'tech_vr', self.length)
        return


    def tech_dpo(self, n=20, m=6):
        """
        区间震荡线指标	dpo(20,6)
        DPO=CLOSE-MA（CLOSE, N/2+1）
        MADPO=MA（DPO,M）
        """
        tech_dpo= self.close - ma(self.close, int(n / 2 + 1))
        tech_dpoma = ma(tech_dpo, m)
        save_hdf(tech_dpo, 'tech_dpo', self.length)
        return


    def tech_trix(self, n=12, m=20):
        """
        三重指数平滑平均	TRIX(12)
        TR= EMA(EMA(EMA(CLOSE,N),N),N)，即进行三次平滑处理
        TRIX=(TR-昨日TR)/ 昨日TR×100
        TRMA=MA（TRIX，M）
        """
        tr = ema(ema(ema(self.close, n), n), n)
        tech_trix = (tr - tr.shift()) / tr.shift() * 100
        tech_trma = ma(tech_trix, m)
        save_hdf(tech_trix, 'tech_trix', self.length)
        return


    def tech_mtm(self, n=6, m=5):
        """
        动力指标	MTM(6,5)
        MTM（N日）=C-REF(C,N)式中，C=当日的收盘价，REF(C,N)=N日前的收盘价；N日是只计算交易日期，剔除掉节假日。
        MTMMA（MTM，N1）= MA（MTM，N1）
        N表示间隔天数，N1表示天数
        """
        tech_mtm = self.close - self.close.shift(n)
        save_hdf(tech_mtm, 'tech_mtm', self.length)
        tech_mtmma = ma(tech_mtm, m)
        save_hdf(tech_mtmma, 'tech_mtmma', self.length)
        return


    def tech_obv(self,):
        """
        能量潮  On Balance Volume
        多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V  # 同花顺貌似用的下面公式
        主公式：当日OBV=前一日OBV+今日成交量
        1.基期OBV值为0，即该股上市的第一天，OBV值为0
        2.若当日收盘价＞上日收盘价，则当日OBV=前一日OBV＋今日成交量
        3.若当日收盘价＜上日收盘价，则当日OBV=前一日OBV－今日成交量
        4.若当日收盘价＝上日收盘价，则当日OBV=前一日OBV
        """
        cond1 = self.close > delay(self.close,1)
        cond2 = self.close < delay(self.close,1)
        cond3 = self.close == delay(self.close,1)
        tech_obv = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        tech_obv[cond1] = tech_obv.expanding(1).sum() + self.volume
        tech_obv[cond2] = tech_obv.expanding(1).sum() - self.volume
        tech_obv[cond3] = tech_obv.expanding(1).sum()
        save_hdf(tech_obv, 'tech_obv', self.length)
        return


    def tech_cci(self, n=14):
        """
        顺势指标
        TYP:=(HIGH+LOW+CLOSE)/3
        CCI:=(TYP-MA(TYP,N))/(0.015×AVEDEV(TYP,N))
        """
        typ = (self.high + self.low + self.close) / 3
        tech_cci = ((typ - typ.rolling(n).mean()) /
                       (0.015 * typ.rolling(min_periods=1, center=False, window=n).apply(
                        lambda x: np.fabs(x - x.mean()).mean())))
        save_hdf(tech_cci, 'tech_cci', self.length)
        return

    def tech_chaikin(self,m=3,n=10):
        """
        AD= VOL x [(CLOSE-LOW) - (HIGH-CLOSE)] / (HIGH - LOW).
        Chaikin Oscillator=EMA(AD,10)-EMA（AD,3）
        Chaikin Volatility=( 10日HIGH-LOW的EMA – 10日前HIGH-LOW的EMA ) / 10
        日前HIGH-LOW的EMA * 100
        :return:
        """
        ad = self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        tech_chaikin = ema(ad,n) - ema(ad,m)
        save_hdf(tech_chaikin, 'tech_chaikin', self.length)
        tech_chaikin_volatility = (ema(self.high-self.low,n) - ema(delay(self.high,10)-delay(self.low,10),n))/ema(delay(self.high,10)-delay(self.low,10),n) *100
        save_hdf(tech_chaikin_volatility, 'tech_chaikin_volatility', self.length)
        return

    def tech_chande(self, n=14):
        """
        CZ1=IF(CLOSE-REF(CLOSE,1)>0,CLOSE-REF(CLOSE,1),0)
        CZ2= IF(CLOSE-REF(CLOSE,1)<0,ABS(CLOSE-REF(CLOSE,1)),0)
        SU=SUM(CZ1,N)
        SD=SUM(CZ2,N)
        CMO=(SU-SD)/(SU+SD)*100
        :return:
        """
        cond1 = self.close>delay(self.close,1)
        cz1 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        cz1[cond1] = self.close-delay(self.close,1)
        cond2= self.close<delay(self.close,1)
        cz2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        cz2[cond2] = abs(self.close-delay(self.close,1))
        su = ts_sum(cz1,n)
        sd = ts_sum(cz2,n)
        tech_chande = (su-sd)/(su+sd)*100
        save_hdf(tech_chande, 'tech_chande', self.length)
        return

    def tech_priceosc(self, n=12, m=26):
        """
        价格振动指数
        PRICEOSC=(MA(C,12)-MA(C,26))/MA(C,12) * 100
        """
        tech_priceosc = (ma(self.close, n) - ma(self.close, m)) / ma(self.close, n) * 100
        save_hdf(tech_priceosc, 'tech_priceosc', self.length)
        return


    def tech_dbcd(self, n=5, m=16, t=76):
        """
        异同离差乖离率	dbcd(5,16,76)
        BIAS=(C-MA(C,N))/MA(C,N)
        DIF=(BIAS-REF(BIAS,M))
        DBCD=SMA(DIF,T,1) =（1-1/T）×SMA(REF(DIF,1),T,1)+ 1/T×DIF
        MM=MA(DBCD,5)
        """
        _bias = (self.close - ma(self.close, n)) / ma(self.close, n)
        _dif = _bias - _bias.shift(m)
        tech_dbcd = sma(_dif, t,1)
        tech_dbcdma = ma(tech_dbcd, n)
        save_hdf(tech_dbcd, 'tech_dbcd', self.length)
        return


    def tech_roc(self, n=12, m=6):
        """
        变动速率	roc(12,6)
        ROC=(今日收盘价-N日前的收盘价)/ N日前的收盘价×100%
        ROCMA=MA（ROC，M）
        ROC:(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)×100
        ROCMA:MA(ROC,M)
        """
        tech_roc = (self.close - self.close.shift(n))/self.close.shift(n) * 100
        tech_rocma = ma(tech_roc, m)
        save_hdf(tech_roc, 'tech_roc', self.length)
        return


    def tech_vroc(self, n=12):
        """
        量变动速率
        VROC=(当日成交量-N日前的成交量)/ N日前的成交量×100%
        """
        tech_vroc = (self.volume - self.volume.shift(n)) / self.volume.shift(n) * 100
        save_hdf(tech_vroc, 'tech_vroc', self.length)
        return


    def tech_cr(self, n=26):
        """ 能量指标
        CR=∑（H-PM）/∑（PM-L）×100
        PM:上一交易日中价（(最高、最低、收盘价的均值)
        H：当天最高价
        L：当天最低价
        """
        pm = delay((self.high + self.low + self.close)/3,1)
        tech_cr = (self.high - pm).rolling(n).sum()/(pm - self.low).rolling(n).sum() * 100
        save_hdf(tech_cr, 'tech_cr', self.length)
        return


    def tech_psy(self, n=12):
        """
        心理指标	PSY(12)
        PSY=N日内上涨天数/N×100
        PSY:COUNT(CLOSE>REF(CLOSE,1),N)/N×100
        MAPSY=PSY的M日简单移动平均
        """
        p = self.close - self.close.shift()
        p[p <= 0] = np.nan
        tech_psy = p.rolling(n).count() / n * 100
        save_hdf(tech_psy, 'tech_psy', self.length)
        return


    def tech_wad(self, n=30):
        """
        威廉聚散指标	WAD(30)
        TRL=昨日收盘价与今日最低价中价格最低者；TRH=昨日收盘价与今日最高价中价格最高者
        如果今日的收盘价>昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRL
        如果今日的收盘价<昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRH
        如果今日的收盘价=昨日的收盘价，则今日的A/D=0
        WAD=今日的A/D+昨日的WAD；MAWAD=WAD的M日简单移动平均
        """
        trl = np.minimum(self.low, self.close.shift(1))
        trh = np.maximum(self.high, self.close.shift(1))
        cond1 = self.close>delay(self.close,1)
        cond2 = self.close<delay(self.close,1)
        ad = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        ad[cond1] = self.close - trl
        ad[cond2] = self.close - trh
        tech_wad = ad.expanding(1).sum()
        tech_wadma = ma(tech_wad, n)
        save_hdf(tech_wad, 'tech_wad', self.length)
        return


    def tech_mfi(self, n=14):
        """
        TYP = (HIGH + LOW + CLOSE)/3
        V1=SUM(IF(TYP>REF(TYP,1),TYP*VOL,0),N)/SUM(IF(TYP<REF(TYP,1),TYP*VOL,0),N)
        MFI=100-(100/ (1+V1))
        """
        typ = (self.high + self.low + self.close)/3
        cond1 = typ > delay(typ,1)
        cond2 = typ < delay(typ,1)
        part1 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1[cond1] = typ*self.volume
        part2[cond2] = typ*self.volume
        v1 = ts_sum(part1,n)/ts_sum(part2,n)
        tech_mfi = 100 * v1 / (1 + v1)  # 同花顺自己给出的公式和实际用的公式不一样，真操蛋，浪费两个小时时间
        save_hdf(tech_mfi, 'tech_mfi', self.length)
        return


    def tech_pvt(self):
        """
        pvt	量价趋势指标
        如果设x=(今日收盘价—昨日收盘价)/昨日收盘价×当日成交量，
        那么当日PVT指标值则为从第一个交易日起每日X值的累加。
        """
        x = (self.close - self.close.shift(1)) / self.close.shift(1) * self.volume
        tech_pvt = x.expanding(1).sum()
        save_hdf(tech_pvt, 'tech_pvt', self.length)
        return


    def tech_wvad(self, n=24, m=6):
        """  # 算法是对的，同花顺计算wvad用的n=6
        威廉变异离散量	wvad(24,6)
        WVAD=N1日的∑ {(当日收盘价－当日开盘价)/(当日最高价－当日最低价)×成交量}
        MAWVAD=MA（WVAD，N2）
        """
        tech_wvad = (np.true_divide((self.close - self.open), (self.high - self.low)) * self.volume).rolling(n).sum()
        tech_wvadma = ma(tech_wvad, m)
        save_hdf(tech_wvad, 'tech_wvad', self.length)
        return


    def tech_cdp(self):
        """
        逆势操作	cdp
        CDP=(最高价+最低价+收盘价)/3  # 同花顺实际用的(H+L+2*c)/4
        AH=CDP+(前日最高价-前日最低价)
        NH=CDP×2-最低价
        NL=CDP×2-最高价
        AL=CDP-(前日最高价-前日最低价)
        """
        tech_cdp = (self.high + self.low + self.close)/3
        tech_cdp_ah = tech_cdp + (self.high.shift(1) - self.low.shift())
        tech_cdp_al = tech_cdp - (self.high.shift(1) - self.low.shift())
        tech_cdp_nh = tech_cdp * 2 - self.low.shift(1)
        tech_cdp_nl = tech_cdp * 2 - self.high.shift(1)
        save_hdf(tech_cdp, 'tech_cdp', self.length)
        return


    def tech_env(self, n=14):
        """
        ENV指标	ENV(14)
        Upper=MA(CLOSE，N)×1.06
        LOWER= MA(CLOSE，N)×0.94
        """
        tech_env_up = mean(self.close,n) * 1.06
        tech_env_down = mean(self.close,n) * 0.94
        save_hdf(tech_env_up, 'tech_env_up', self.length)
        save_hdf(tech_env_down, 'tech_env_down', self.length)
        return


    def tech_mike(self, n=12):
        """
        麦克指标	mike(12)
        初始价（TYP）=（当日最高价＋当日最低价＋当日收盘价）/3
        HV=N日内区间最高价
        LV=N日内区间最低价
        初级压力线（WR）=TYP×2-LV
        中级压力线（MR）=TYP+HV-LV
        强力压力线（SR）=2×HV-LV
        初级支撑线（WS）=TYP×2-HV
        中级支撑线（MS）=TYP-HV+LV
        强力支撑线（SS）=2×LV-HV
        """
        typ = (self.high+self.low+self.close)/3
        hv = self.high.rolling(n).max()
        lv = self.low.rolling(n).min()
        tech_mike_wr = typ * 2 - lv
        tech_mike_mr = typ + hv - lv
        tech_mike_sr = 2 * hv - lv
        tech_mike_ws = typ * 2 - hv
        tech_mike_ms = typ - hv + lv
        tech_mike_ss = 2 * lv - hv
        save_hdf(tech_mike_wr, 'tech_mike_wr', self.length)
        save_hdf(tech_mike_mr, 'tech_mike_mr', self.length)
        save_hdf(tech_mike_sr, 'tech_mike_sr', self.length)
        save_hdf(tech_mike_ws, 'tech_mike_ws', self.length)
        save_hdf(tech_mike_ms, 'tech_mike_ms', self.length)
        save_hdf(tech_mike_ss, 'tech_mike_ss', self.length)
        return


    def tech_vma(self, n=5):
        """
        量简单移动平均	VMA(5)	VMA=MA(volume,N)
        VOLUME表示成交量；N表示天数
        """
        tech_vma = ma(self.volume, n)
        save_hdf(tech_vma, 'tech_vma', self.length)
        return


    def tech_vmacd(self, qn=12, sn=26, m=9):
        """
        量指数平滑异同平均	vmacd(12,26,9)
        今日EMA（N）=2/（N+1）×今日成交量+(N-1)/（N+1）×昨日EMA（N）
        DIFF= EMA（N1）- EMA（N2）
        DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)
        MACD（BAR）=2×（DIF-DEA）
        """
        diff = ema(self.volume, qn) - ema(self.volume, sn)
        dea = ema(diff, m)
        tech_vmacd = 2*(diff- dea)
        save_hdf(tech_vmacd, 'tech_vmacd', self.length)
        return


    def tech_vosc(self, n=12, m=26):
        """
        成交量震荡	vosc(12,26)
        VOSC=（MA（VOLUME,SHORT）- MA（VOLUME,LONG））/MA（VOLUME,SHORT）×100
        """
        tech_vosc = (ma(self.volume, n) - ma(self.volume, m)) / ma(self.volume, n) * 100
        save_hdf(tech_vosc, 'tech_vosc', self.length)
        return


    def tech_tapi(self, n=6):
        """ #
        加权指数成交值	tapi(6)
        TAPI=每日成交总值/当日加权指数=a/PI；A表示每日的成交金额，PI表示当天的股价指数即指收盘价
        """
        tech_tapi = self.amount / self.close
        tech_tapima = ma(tech_tapi, n)
        save_hdf(tech_tapi, 'tech_tapi', self.length)
        return


    def tech_vstd(self, n=10):
        """
        成交量标准差	vstd(10)
        VSTD=STD（Volume,N）=[∑（Volume-MA(Volume，N)）^2/N]^0.5
        """
        tech_vstd = self.volume.rolling(n).std(ddof=1)
        save_hdf(tech_vstd, 'tech_vstd', self.length)
        return


    def tech_mi(self, n=12):
        """
        动量指标	mi(12)
        A=CLOSE-REF(CLOSE,N)
        MI=SMA(A,N,1)
        """
        tech_mi = sma(self.close - self.close.shift(n), n, 1)
        save_hdf(tech_mi, 'tech_mi', self.length)
        return


    def tech_micd(self, n=3, m=10, k=20):
        """
        异同离差动力指数	micd(3,10,20)
        MI=CLOSE-ref(CLOSE,1)AMI=SMA(MI,N1,1)
        DIF=MA(ref(AMI,1),N2)-MA(ref(AMI,1),N3)
        MICD=SMA(DIF,10,1)
        """
        mi = self.close - self.close.shift(1)
        ami = sma(mi, n, 1)
        dif = ma(ami.shift(1), m) - ma(ami.shift(1), k)
        tech_micd= sma(dif, m, 1)
        save_hdf(tech_micd, 'tech_micd', self.length)
        return


    def tech_rc(self, n=50):
        """
        变化率指数	rc(50)
        RC=收盘价/REF（收盘价，N）×100
        ARC=EMA（REF（RC，1），N，1）
        """
        tech_rc = self.close / self.close.shift(n) * 100
        tech_arc = ema(delay(tech_rc,1), n)
        save_hdf(tech_arc, 'tech_arc', self.length)
        return


    def tech_rccd(self, n=59, m=21, k=28):
        """  #
        异同离差变化率指数 rate of change convergence divergence	rccd(59,21,28)
        RC=收盘价/REF（收盘价，N）×100%
        ARC=EMA(REF(RC,1),N,1)
        DIF=MA(ref(ARC,1),N1)-MA MA(ref(ARC,1),N2)
        RCCD=SMA(DIF,N,1)
        """
        rc = self.close / delay(self.close,n) * 100
        arc = ema(delay(rc,1), n)
        tech_rccd = sma(ma(arc.shift(), m) - ma(arc.shift(), k), n, 1)
        save_hdf(tech_rccd, 'tech_rccd', self.length)
        return


    def tech_srmi(self, n=9):
        """
        SRMIMI修正指标	srmi(9)
        如果收盘价>N日前的收盘价，SRMI就等于（收盘价-N日前的收盘价）/收盘价
        如果收盘价<N日前的收盘价，SRMI就等于（收盘价-N日签的收盘价）/N日前的收盘价
        如果收盘价=N日前的收盘价，SRMI就等于0
        """
        cond1 = self.close>delay(self.close,n)
        cond2 = self.close<delay(self.close,n)
        tech_srmi = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        tech_srmi[cond1] = (self.close-delay(self.close,n))/self.close
        tech_srmi[cond2] = (self.close-delay(self.close,n))/delay(self.close,n)
        save_hdf(tech_srmi, 'tech_srmi', self.length)
        return


    def tech_dptb(self, n=7):
        """
        大盘同步指标	dptb(7)
        DPTB=（统计N天中个股收盘价>开盘价，且指数收盘价>开盘价的天数或者个股收盘价<开盘价，且指数收盘价<开盘价）/N
        """
        cond1 = (self.close > self.open) & (pd.DataFrame(np.tile((self.benchmark_close > self.benchmark_open), (self.close.shape[1], 1)),index=self.close.columns, columns=self.close.index).T)
        cond2 = (self.close < self.open) & (pd.DataFrame(np.tile((self.benchmark_close < self.benchmark_open), (self.close.shape[1], 1)),index=self.close.columns, columns=self.close.index).T)
        part1 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1[cond1|cond2]=1
        tech_dptb = count(part1,n)/n
        save_hdf(tech_dptb, 'tech_dptb', self.length)
        return


    def tech_jdqs(self, n=20):
        """
        阶段强势指标	jdqs(20)
        JDQS=（统计N天中个股收盘价>开盘价，且指数收盘价<开盘价的天数）/（统计N天中指数收盘价<开盘价的天数）
        """
        cond1 = (self.close > self.open) & (pd.DataFrame(np.tile((self.benchmark_close < self.benchmark_open), (self.close.shape[1], 1)),index=self.close.columns, columns=self.close.index).T)
        cond2 = self.benchmark_close < self.benchmark_open
        part1 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        part2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part2[cond2] = 1
        tech_jdqs = count(part1, n) / count(part2, n)
        save_hdf(tech_jdqs, 'tech_jdqs', self.length)
        return


    def tech_jdrs(self, n=20):
        """
        阶段弱势指标	jdrs(20)
        JDRS=（统计N天中个股收盘价<开盘价，且指数收盘价>开盘价的天数）/（统计N天中指数收盘价>开盘价的天数）
        """
        cond1 = (self.close < self.open) & (pd.DataFrame(np.tile((self.benchmark_close > self.benchmark_open), (self.close.shape[1], 1)),index=self.close.columns, columns=self.close.index).T)
        cond2 = self.benchmark_close > self.benchmark_open
        part1 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1[cond1] = 1
        part2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        part2[cond2] = 1
        tech_jdrs = ts_sum(part1,n)/ts_sum(part2,n)
        save_hdf(tech_jdrs, 'tech_jdrs', self.length)
        return


    def tech_zdzb(self, n=125, m=5, k=20):
        """
        筑底指标	zdzb(125,5,20)
        A=（统计N1日内收盘价>=前收盘价的天数）/（统计N1日内收盘价<前收盘价的天数）
        B=MA（A,N2）
        D=MA（A，N3）
        """
        p = self.close - delay(self.close,1)
        q = p.copy()
        p[p < 0] = np.nan
        q[q >= 0] = np.nan
        tech_zdzb_a= p.rolling(n).count() / q.rolling(n).count()
        tech_zdzb_b = mean(tech_zdzb_a,m)
        tech_zdzb_d = mean(tech_zdzb_a,k)
        save_hdf(tech_zdzb_a, 'tech_zdzb_a', self.length)
        return


    def tech_mass(self, n=9, m=25):
        """
        梅丝线	mass(9,25)
        AHL=MA(（H-L）,N1)
        BHL= MA（AHL，N1）
        MASS=SUM（AHL/BHL，N2）
        H：表示最高价；L：表示最低价
        """
        tech_mass = ts_sum(ma((self.high - self.low), n) / ma(ma((self.high - self.low), n), n),m)
        save_hdf(tech_mass, 'tech_mass', self.length)
        return


    def tech_vhf(self, n=28):
        """
        纵横指标	vhf(28)
        VHF=（N日内最大收盘价与N日内最小收盘价之前的差）/（N日收盘价与前收盘价差的绝对值之和）
        """
        tech_vhf = (ts_max(self.close,n)- ts_min(self.close,n)) / ts_sum(abs(self.close - delay(self.close,1)),n)
        save_hdf(tech_vhf, 'tech_vhf', self.length)
        return


    def tech_cvlt(self, n=10):
        """
        佳庆离散指标	cvlt(10)
        cvlt=（最高价与最低价的差的指数移动平均-前N日的最高价与最低价的差的指数移动平均）/前N日的最高价与最低价的差的指数移动平均
        """
        p = ema((delay(self.high,n) - delay(self.low,n)), n)
        tech_cvlt = (ema(self.high - self.low, n) - p) / p * 100
        save_hdf(tech_cvlt, 'tech_cvlt', self.length)
        return


    def tech_up_n(self):
        """
        连续上涨天数，当天收盘价大于开盘价即为上涨一天 # 同花顺实际结果用收盘价-前一天收盘价
        """
        p = (self.close - delay(self.close,1)).fillna(0)
        p[p > 0] = 1
        p[p < 0] = 0
        tech_up_n = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        for index, col in p.iteritems():
            tech_up_n[index] = up_n(col)
        save_hdf(tech_up_n, 'tech_up_n', self.length)
        return


    def tech_down_n(self):
        """
        连续下跌天数，当天收盘价小于开盘价即为下跌一天 # 同花顺实际结果用收盘价-前一天收盘价
        """
        p = (self.close - delay(self.close,1)).fillna(0)
        p[p > 0] = 0
        p[p < 0] = 1
        tech_down_n = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        for index, col in p.iteritems():
            tech_down_n[index] = up_n(col)
        save_hdf(tech_down_n, 'tech_down_n', self.length)
        return


def run_func(paras):
    startdate = paras["startdate"]
    enddate = paras["enddate"]
    count = paras["count"]
    length = paras["length"]
    techs = TechFactor(startdate,enddate, count, length)
    func_list = paras["func"]
    for func_name in func_list:
        eval("techs." + func_name + "()")
    return


def set_params(func_list, Start_Date, End_Date , count, length):
    td = {"module_name": "TechFactor",
          "startdate":Start_Date,
          "enddate":End_Date,
          "count":count,
          "length":length}
    params = []
    for i, sec_code in enumerate(func_list):
        td['func'] = sec_code
        params.append(td.copy())
    return params


if __name__ == "__main__":
    """设置更新日期"""
    TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
    Start_Date = None
    End_Date = str(TradeDate.iloc[-1, 0])
    count = 600  # 用于计算因子的数据长度
    length = TradeDate.shape[0]  # 输出更新的数据长度
    techs = TechFactor(Start_Date, End_Date, count, length)
    func_list = [x for x in dir(techs) if x.startswith("t")]
    func_list = [func_list[i:i + 1] for i in range(0, len(func_list), 1)]
    paras = set_params(func_list, Start_Date, End_Date , count, length)
    pool = multiprocessing.Pool(12)
    pool.map(run_func, paras)
    pool.close()
    pool.join()
    print("The data of Techfactor update completely!")
