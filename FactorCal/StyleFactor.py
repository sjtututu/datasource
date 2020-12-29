#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/5 21:54
# @Author: Tu Xinglong
# @File  : StyleFactor.py

import os
import sys

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
data_path = Pre_path + '\\FactorData\\FinaFactor\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa


class StyleFactor(object):
	'''
	Barra中九大风格因子
	'''
	def __init__(self, startdate, enddate, count, length):
		'''
        获取数据信息
        enddate为必选参数，startdate和count二选一
        有startdate则数据区间为startdate-enddate
        无startdate则数据以enddate截止，数量为count
        length:控制结果输出长度
        '''
		stock_price = get_price(startdate=startdate,
								enddate=enddate,
								fields=['close_qfq','ret','turnover_rate'],
								count=count)
		benchmark_price = get_price(startdate=startdate,
									enddate=enddate,
									fields=['index'],
									count=count)
		fina_price = get_price(startdate=startdate,
									enddate=enddate,
									fields=['pe_ttm','eps_FY1','cfps','cagrgr_PY5','cagrpni_PY5',
											'sestni_YOY1','sestni_YOY3','total_assets','total_equity',
											'total_liab','total_mv','total_ncl'],
									count=count)
		# 百分比统一用小数表示，价量单位统一用元
		self.close         = stock_price['close_qfq']
		self.returns       = stock_price['ret']
		self.turnover_rate = stock_price['turnover_rate']/100
		self.benchmark_ret = benchmark_price['index']['沪深300close'].pct_change().fillna(method='ffill').fillna(0).replace([np.inf,-np.inf],[0,0])
		self.pe_ttm        = fina_price['pe_ttm']
		self.eps_FY1       = fina_price['eps_FY1']
		self.cfps          = fina_price['cfps']
		self.cagrgr_PY5    = fina_price['cagrgr_PY5']
		self.cagrpni_PY5   = fina_price['cagrpni_PY5']
		self.sestni_YOY1   = fina_price['sestni_YOY1']
		self.sestni_YOY3   = fina_price['sestni_YOY3']
		self.total_assets  = fina_price['total_assets']
		self.total_equity  = fina_price['total_equity']
		self.total_liab    = fina_price['total_liab']
		self.total_ncl     = fina_price['total_ncl']
		self.total_mv      = fina_price['total_mv'] * 10000
		self.length        = length
		np.seterr(divide='ignore', invalid='ignore')  # 忽略警告
	"""
	1.用原始数据计算barra因子时要填充(前值填充)
	2.小类因子合成大类因子时要先去极值和标准化(因子截面)
	3.大类因子计算完之后再进行去极值和标准化(因子截面)
	4.三倍标准差去极值，标准化沿因子截面方向
	"""

	def beta(self, window = 250):
		w=(0.5**(np.array(list(range(1,251,1)))/60))[::-1]
		rows=self.returns.shape[0]
		columns = self.returns.columns
		beta = np.ones(self.returns.shape) * np.nan
		e = np.ones(self.returns.shape) * np.nan
		for i,col in enumerate(columns):
			print(col)
			for j in range(window-1,rows):
				model=OLS(x=np.array(self.benchmark_ret[j - window + 1:j + 1])*w,
						  y=np.array(self.returns[col][j - window + 1:j + 1])*w)
				beta[j,i] = model.beta
				e[j,i] = model.std_err
		beta = pd.DataFrame(beta,index=self.returns.index,columns=self.returns.columns)#.iloc[-self.length:,:]
		beta = beta.replace([np.inf,-np.inf],[0,0]).fillna(method="ffill").fillna(0) # 填充
		beta = remove_extreme(beta,axis=1)      # 三倍标准差去极值
		beta = pd.DataFrame(preprocessing.scale(beta,axis=1), index=self.returns.index,
							columns=self.returns.columns) # 因子截面方向标准化，均值0方差1
		beta.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\beta.h5","beta",mode="a",format="table",append=True)
		e = pd.DataFrame(e,index=self.returns.index,columns=self.returns.columns)#.iloc[-self.length:,:]
		return e


	def momentum(self,window=520):
		w = np.reshape((0.5 ** (np.array(list(range(1, 501, 1))) / 120))[::-1], (500, 1))
		rstr = np.ones(self.returns.shape) * np.nan
		rows = self.returns.shape[0]
		for i in range(window-1,rows):
			rstr[i,] = np.sum(np.log(np.array(self.returns.iloc[i-window+1:i-20+1])+1) * w,axis=0)
		momentum = pd.DataFrame(rstr,index=self.returns.index,columns=self.returns.columns)
		momentum = momentum.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)
		momentum = remove_extreme(momentum, axis=1)  # 三倍标准差去极值
		momentum = pd.DataFrame(preprocessing.scale(momentum, axis=1), index=self.returns.index,
							columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
		momentum.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\momentum.h5", "momentum", mode="a", format="table",append=True)
		return


	def size(self):
		lncap = np.log(self.total_mv)
		size = pd.DataFrame(lncap,index=self.returns.index,columns=self.returns.columns)#.iloc[-self.length:,:]
		size = size.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		size = remove_extreme(size, axis=1)  # 三倍标准差去极值
		size = pd.DataFrame(preprocessing.scale(size, axis=1), index=self.returns.index,
								columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
		size.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\size.h5", "size", mode="a", format="table",append=True)
		return


	def earningsyield(self):
		epibs = np.divide(self.eps_FY1,self.close)
		epibs = epibs.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		epibs = remove_extreme(epibs, axis=1)  # 三倍标准差去极值
		epibs = pd.DataFrame(preprocessing.scale(epibs, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		#==============================================================
		etop = np.divide(1,self.pe_ttm)
		etop = etop.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		etop = remove_extreme(etop, axis=1)  # 三倍标准差去极值
		etop = pd.DataFrame(preprocessing.scale(etop, axis=1), index=self.returns.index,
							 columns=self.returns.columns)
		#==============================================================
		cetop = np.divide(self.cfps,self.close)
		cetop = cetop.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		cetop = remove_extreme(cetop, axis=1)  # 三倍标准差去极值
		cetop = pd.DataFrame(preprocessing.scale(cetop, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		#==============================================================
		earningsyield = 0.68 * epibs + 0.11 * etop + 0.21 * cetop
		earningsyield = earningsyield.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		earningsyield = remove_extreme(earningsyield, axis=1)  # 三倍标准差去极值
		earningsyield = pd.DataFrame(preprocessing.scale(earningsyield, axis=1), index=self.returns.index,
							 columns=self.returns.columns)
		earningsyield.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\earningsyield.h5", "earningsyield", mode="a", format="table",append=True)
		return

	# 波动率因子
	def volatility(self, e, window=250):
		w=np.reshape((0.5**(np.array(list(range(1,251,1)))/40))[::-1],(250,1))
		dastd = np.ones(self.returns.shape) * np.nan
		rows = self.returns.shape[0]
		for i in range(window - 1, rows):
			dastd[i,] = pow(np.sum(pow((np.array(self.returns.iloc[i - window + 1:i + 1])-np.array(self.returns.iloc[i - window + 1:i + 1]).mean(axis=0)),2) * w, axis=0),1/2)
		dastd = pd.DataFrame(dastd, index=self.returns.index, columns=self.returns.columns)  # .iloc[-self.length:,:]
		dastd = dastd.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		dastd = remove_extreme(dastd, axis=1)  # 三倍标准差去极值
		dastd = pd.DataFrame(preprocessing.scale(dastd, axis=1), index=self.returns.index,
							columns=self.returns.columns)  # 因子截面方向标准化，均值0方差1
		#=============================================================
		cmra = np.ones(self.returns.shape) * np.nan
		for i in range(window + 2, rows):
			temp = []  # 存储12个月收益率,每月以21天计算
			for m in [12,11,10,9,8,7,6,5,4,3,2,1]:
				month_return = []  # 单月收益率
				a = self.returns.iloc[i - 21 * m:i - 21* (m - 1),:]
				month_return = np.log(np.sum(self.returns.iloc[i - 21 * m:i - 21* (m - 1), :], axis=0) + 1)
				temp.append(month_return)
			temp = np.cumsum(np.array(temp), axis=0)
			cmra[i,] = np.log(1 + np.max(temp, axis=0)) - np.log(1 + np.min(temp, axis=0))
		cmra = pd.DataFrame(cmra, index=self.returns.index, columns=self.returns.columns)
		cmra = cmra.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)
		cmra = remove_extreme(cmra, axis=1)
		cmra = pd.DataFrame(preprocessing.scale(cmra, axis=1), index=self.returns.index,
							 columns=self.returns.columns)
		#==============================================================
		hsigma = e
		e = e.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)
		e = remove_extreme(e, axis=1)
		e = pd.DataFrame(preprocessing.scale(e, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		#==============================================================
		volatility = 0.74*dastd + 0.16*cmra + 0.1*hsigma
		volatility = volatility.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		volatility = remove_extreme(volatility, axis=1)  # 三倍标准差去极值
		volatility = pd.DataFrame(preprocessing.scale(volatility, axis=1), index=self.returns.index,
									 columns=self.returns.columns)
		volatility.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\volatility.h5", "volatility", mode="a", format="table",append=True)
		return

	def growth(self):
		sgro = self.cagrgr_PY5
		sgro = sgro.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		sgro = remove_extreme(sgro, axis=1)  # 三倍标准差去极值
		sgro = pd.DataFrame(preprocessing.scale(sgro, axis=1), index=self.returns.index,
									 columns=self.returns.columns)
		#=============================================================
		egro = self.cagrpni_PY5
		egro = egro.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		egro = remove_extreme(egro, axis=1)  # 三倍标准差去极值
		egro = pd.DataFrame(preprocessing.scale(egro, axis=1), index=self.returns.index,
								  columns=self.returns.columns)
		#=============================================================
		egib = self.sestni_YOY3
		egib = egib.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		egib = remove_extreme(egib, axis=1)  # 三倍标准差去极值
		egib = pd.DataFrame(preprocessing.scale(egib, axis=1), index=self.returns.index,
								  columns=self.returns.columns)
		#=============================================================
		egib_s = self.sestni_YOY1
		egib_s = egib_s.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		egib_s = remove_extreme(egib_s, axis=1)  # 三倍标准差去极值
		egib_s = pd.DataFrame(preprocessing.scale(egib_s, axis=1), index=self.returns.index,
								  columns=self.returns.columns)
		#=============================================================
		growth = 0.47*sgro+0.24*egro+0.18*egib+0.11*egib_s
		growth = growth.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		growth = remove_extreme(growth, axis=1)  # 三倍标准差去极值
		growth = pd.DataFrame(preprocessing.scale(growth, axis=1), index=self.returns.index,
								  columns=self.returns.columns)
		growth.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\growth.h5", "growth", mode="a", format="table",append=True)
		return

	def value(self):
		btop = np.divide(self.total_equity,self.total_mv)
		value = btop.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		value = remove_extreme(value, axis=1)  # 三倍标准差去极值
		value = pd.DataFrame(preprocessing.scale(value, axis=1), index=self.returns.index,
							  columns=self.returns.columns)
		value.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\value.h5", "value", mode="a", format="table",append=True)
		return

	def leverage(self):
		mlev = np.divide((self.total_mv+self.total_ncl),self.total_mv)
		mlev = mlev.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		mlev = remove_extreme(mlev, axis=1)  # 三倍标准差去极值
		mlev = pd.DataFrame(preprocessing.scale(mlev, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		#===============================================================
		dtoa = np.divide(self.total_liab,self.total_assets)
		dtoa = dtoa.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		dtoa = remove_extreme(dtoa, axis=1)  # 三倍标准差去极值
		dtoa = pd.DataFrame(preprocessing.scale(dtoa, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		# ===============================================================
		blev = np.divide((self.total_equity+self.total_ncl),self.total_equity)
		blev = blev.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		blev = remove_extreme(blev, axis=1)  # 三倍标准差去极值
		blev = pd.DataFrame(preprocessing.scale(blev, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		# ===============================================================
		leverage = 0.38*mlev+0.35*dtoa+0.27*blev
		leverage = leverage.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		leverage = remove_extreme(leverage, axis=1)  # 三倍标准差去极值
		leverage = pd.DataFrame(preprocessing.scale(leverage, axis=1), index=self.returns.index,
							  columns=self.returns.columns)
		leverage.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\leverage.h5", "leverage", mode="a", format="table",append=True)
		return 

	def liquidity(self):
		stom = np.log(ts_sum(self.turnover_rate,21))
		stom = stom.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		stom = remove_extreme(stom, axis=1)  # 三倍标准差去极值
		stom = pd.DataFrame(preprocessing.scale(stom, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		# ===============================================================
		stoq = np.log((1/3)*(ts_sum(self.turnover_rate,63)))
		stoq = stoq.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		stoq = remove_extreme(stoq, axis=1)  # 三倍标准差去极值
		stoq = pd.DataFrame(preprocessing.scale(stoq, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		# ===============================================================
		stoa = np.log((1/12)*(ts_sum(self.turnover_rate,252)))
		stoa = stoa.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		stoa = remove_extreme(stoa, axis=1)  # 三倍标准差去极值
		stoa = pd.DataFrame(preprocessing.scale(stoa, axis=1), index=self.returns.index,
							columns=self.returns.columns)
		# ===============================================================
		liquidity = 0.35*stom + 0.35*stoq + 0.3 *stoa
		liquidity = liquidity.replace([np.inf, -np.inf], [0, 0]).fillna(method="ffill").fillna(0)  # 填充
		liquidity = remove_extreme(liquidity, axis=1)  # 三倍标准差去极值
		liquidity = pd.DataFrame(preprocessing.scale(liquidity, axis=1), index=self.returns.index,
								columns=self.returns.columns)
		liquidity.iloc[-self.length:].to_hdf(FactorPath + "StyleFactor\\liquidity.h5", "liquidity", mode="a", format="table",append=True)
		return


if __name__ == '__main__':
	"""设置更新日期"""
	pro = ts.pro_api()
	TradeDate = pd.read_csv(Pre_path + "\\TradeDate.csv")
	Start_Date = None
	End_Date = str(TradeDate.iloc[-1, 0])
	count = 600  # 用于计算因子的数据长度
	length = TradeDate.shape[0]  # 输出更新的数据长度
	styles = StyleFactor(Start_Date, End_Date, count, length)
	e = styles.beta()
	styles.momentum()
	styles.size()
	styles.earningsyield()
	styles.volatility(e)
	styles.growth()
	styles.value()
	styles.leverage()
	styles.liquidity()
	print("The data of StyleFactor update completely!")
