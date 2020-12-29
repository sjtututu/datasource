#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

import itertools
import talib as tl
import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from pyfinance.ols import OLS, PandasRollingOLS

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'


def get_stock(date, ST=False, Suspend=False):
    """
    根据日期索引股票池代码
    默认读取全A非停牌，非ST股票
    ST : True为保留ST股票池
    Suspend: True为保留停牌股票池
    """
    data_path = "E:\\AlphaPlatform\\StockRef\\"
    stock_a = pd.read_hdf(data_path + "Stock_A\\" + date + ".h5", date).SecuCode.tolist()
    if not ST:
        stock_st = pd.read_hdf(data_path + "Stock_ST\\" + date + ".h5", date).SecuCode.tolist()
    else:
        stock_st = []
    if not Suspend:
        stock_suspend = pd.read_hdf(data_path + "Stock_Suspend\\" + date + ".h5", date).SecuCode.tolist()
    else:
        stock_suspend = []
    stock_list  = [x for x in stock_a if x not in stock_st and x not in stock_suspend]
    return stock_list


def get_price(security='stock', startdate=None, enddate=None,
              fields=None, count=250):
    '''
    获取历史数据
    :param security: 股票代码,all代表取所有股票
    :param start_date: 数据开始日期
    :param end_date: 数据截止日期
    :param frequency: 数据频率
    :param fields: 获取的数据种类
    :param fq: 复权方式
    :param count：数据长度,默认取一
    :return: list
    '''
    df = {}
    for index, factor_name in enumerate(fields):
        df[factor_name] = pd.read_hdf(FactorPath + "FinaFactor\\" + factor_name + ".h5", factor_name)
        df[factor_name].index = df[factor_name].index.astype("int64")
        # 有起始日期则按照起始日期取数据
        # 没有的话则按照数据长度取数据
        if startdate is None:
            trade_date = [str(x) for x in df[factor_name].index]
            index = trade_date.index(enddate)
            start_date = trade_date[index-count]
            end_date = enddate
            df[factor_name] = df[factor_name].ix[start_date:end_date, :]
        else:
            df[factor_name] = df[factor_name].ix[startdate:enddate, :]

    return df

def save_h5(factor_data, date):
    """
    因子数据按照股票，日期方式存储
    """
    data_path = "D:\\HistoryData\\FactorData\\AlphaFactors\\"
    factor_data = pd.DataFrame(factor_data).T
    factor_data = factor_data.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
    factor_data = remove_extreme(factor_data, axis=1)  # 三倍标准差去极值
    for col, value in factor_data.iteritems():
        if not os.path.isdir(data_path + col):
            os.mkdir(data_path + col)
            factor_data[col].to_hdf(data_path + col + "\\" + date + ".h5", date)
        else:
            factor_data[col].to_hdf(data_path + col + "\\" + date + ".h5", date)
    return

def save_hdf(factordata, filename, length=1):
    Index = factordata.index
    Columns = factordata.columns
    factordata = factordata.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
    factordata = remove_extreme(factordata, axis=1)  # 三倍标准差去极值
    factordata = pd.DataFrame(preprocessing.scale(factordata, axis=1), index=Index,columns=Columns).iloc[-length:]
    if filename[-3:] == 'err':
        factordata.to_hdf(FactorPath + "ErrorFactor\\" + filename + ".h5", filename, mode="a", format="table",append=True)
    elif filename[0:4] == 'alph':
        factordata.to_hdf(FactorPath + "Alpha101\\" + filename + ".h5",filename , mode="a", format="table",append=True)
    elif filename[0:5] == 'tech_':
        factordata.to_hdf(FactorPath + "TechFactor\\" + filename +".h5",filename , mode="a", format="table",append=True)
    elif filename[0:4] == 'fina':
        factordata.to_hdf(FactorPath + "FinaFactor\\" + filename +".h5",filename , mode="a", format="table",append=True)
    else:
        factordata.to_hdf(FactorPath + "Tech191\\" + filename + ".h5",filename , mode="a", format="table",append=True)
    return

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def mean(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()


def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # return df.rank(axis=1, pct=True)
    return df.rank(axis=1,pct=True)


def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.as_matrix()
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1:row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)


def sma(df, m, n):
    """
    平滑移动指标 Smooth Moving Average
    """
    return df.ewm(min_periods=0, ignore_na=False, adjust=False, alpha=n/m).mean()


def _wma(df, window):
    """
    :param df:
    :param window:
    :return:
    """
    return tl.WMA(np.array(df), window)[-1]


def wma(df, window=10):
    """

    :param df:
    :param window:
    :return:
    """
    return df.rolling(window).apply(_wma, args=([window]))


def sequence(n):
    """
    生成1~n的等差序列
    :param n:
    :return:
    """
    return pd.Series(np.arange(1, n+1))


def ols(x, y):
    """
    面板数据线性回归
    :param x:
    :param y:
    :return:
    """
    linger = LinearRegression()
    module = linger.fit(x, y)

    return module.coef_, module.intercept_


def regbeta(df, seq, window=10):
    """
    线性回归系数
    :param x:
    :param y:
    :param n:
    :return:
    """
    df = df.fillna(method='ffill').fillna(0)
    rows = df.shape[0]
    columns = df.shape[1]
    beta = np.zeros([rows, columns])
    beta[0:window - 1] = np.nan
    alpha = np.zeros([rows, columns])
    alpha[0:window - 1] = np.nan
    for i in range(window - 1, rows):
        a, b = ols(df.iloc[i - window + 1:i + 1], seq)
        beta[i, :] = a
        alpha[i, :] = b
    beta = pd.DataFrame(beta, index=df.index, columns=df.columns)
    return beta


def count(cond, window):
    """
    :param df:
    :param window:
    :return:
    """
    return cond.rolling(window).sum()


def _highday(na):
    """
    :param na:
    :return:
    """
    return len(na) - np.argmax(na) - 1


def highday(df, window=10):
    """
    :param df:
    :param window:
    :return:
    """
    return df.rolling(window).apply(_highday)


def _lowday(na):
    """
    :param na:
    :return:
    """
    return len(na) - np.argmin(na) - 1


def lowday(df, window=10):
    """
    :param df:
    :param window:
    :return:
    """
    return df.rolling(window).apply(_lowday)


# ==============================================================
# ==============================================================


def ma(df, n=10):
    """
    移动平均线 Moving Average
    MA（N）=（第1日收盘价+第2日收盘价—+……+第N日收盘价）/N
    """
    return df.rolling(n).mean()


def _ma(series, n):
    """
    移动平均
    """
    return series.rolling(n).mean()


def md(df, n=10):
    """
    移动标准差
    STD=S（CLOSE,N）=[∑（CLOSE-MA(CLOSE，N)）^2/N]^0.5
    """
    return df.rolling(n).std(ddof=0)


def _md(series, n):
    """
    标准差MD
    """
    return series.rolling(n).std(ddof=0)


def ema(df, n=12):
    """
    指数平均数指标 Exponential Moving Average
    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）
    EMA(X,N)=[2×X+(N-1)×EMA(ref(X),N]/(N+1)
    """
    return df.ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()


def _ema(series, n):
    """
    指数平均数
    """
    return series.ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()


def up_n(Series):
    """
    连续上涨天数，当天收盘价大于开盘价即为上涨一天 # 同花顺实际结果用收盘价-前一天收盘价
    """
    m = []
    for k, g in itertools.groupby(Series):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    return m

def remove_extreme(df,axis=1):
    ma = np.mean(df,axis=1)
    std = np.std(df,axis=1)
    sigma_up = pd.DataFrame(np.tile(np.array(ma + 3 * std),(df.shape[1],1)),index=df.columns,columns=df.index).T
    sigma_down = pd.DataFrame(np.tile(np.array(ma - 3 * std),(df.shape[1],1)),index=df.columns,columns=df.index).T
    df[df > sigma_up] = sigma_up
    df[df < sigma_down] = sigma_down
    return df


if __name__ == '__main__':
    get_price(enddate='20190426',
              fields=[
                  'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq', 'volume',
                  'amount', 'ret'
              ],
              count=1000)
    print()
