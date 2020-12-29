#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/7/30 11:00
# @Author: Tu Xinglong
# @File  : F_SetUp.py

import os
import sys

import datetime as dt
import pandas as pd

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
Pre_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path += [CurrentPath, Pre_path, Pre_path + '\\Engine']
FactorPath = Pre_path + '\\FactorData\\'


def set_params(file_list):
    new_file_list = []
    for file_name in file_list:
        new_file_list.append("python " + CurrentPath + "\\" + file_name)
    return new_file_list

file_list = ['Alpha101.py','Tech191.py','TechFactor.py',
             'IndusFactor.py','StyleFactor.py','ErrorFactor.py'] # FinaFactor.py
new_file_list = set_params(file_list)
for pyfile in new_file_list:
    os.system(pyfile)
print("Factor files update completely!")