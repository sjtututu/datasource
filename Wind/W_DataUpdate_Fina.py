#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/9 8:29
# @Author: Tu Xinglong
# @File  : W_DataUpdate.py

import os
import sys

import pandas as pd
from dateutil.parser import parse

from WindPy import *  # noqa E403
w.start()  # noqa E405

CurrentPath = os.getcwd()
Pre_path = os.path.dirname(os.getcwd())
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
DataPath = Pre_path + '\\MarketInfo\\'
"""
日期设定,以3天为一个周期进行更新
end_date最后一天只能是交易日
注：日期都填YYYYMMDD格式，比如20181010
"""
start_date = '2019-02-01'
end_date = '2019-02-28'
report_date = '2018-09-30'  # 报告期记得注意修改
