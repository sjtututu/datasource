#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/5 21:56
# @Author: Tu Xinglong
# @File  : FinaFactor.py

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
import matplotlib.pyplot as plt


CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'

from TechFunc import *  # noqa
from LoggingPlus import Logger  # noqa


class FinaFactor(object):
    '''
    基本面财务因子
    '''
    def __init__(self, startdate, enddate, count, length):
        pass

