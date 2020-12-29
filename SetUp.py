#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/9 14:40
# @Author: Tu Xinglong
# @File  : SetUp.py

import os
import sys
import multiprocessing
import tushare as ts
import datetime as dt
import pandas as pd

CurrentPath = os.path.abspath(os.path.dirname(__file__))  # 设置绝对路径
sys.path += [
    CurrentPath, CurrentPath + '\\Engine', CurrentPath + "\\Choice",
    CurrentPath + "\\Wind", CurrentPath + "\\Tushare"
]
T_filepath = CurrentPath + "\\Tushare\\"
# W_filepath = CurrentPath + "\\Wind\\"
C_filepath = CurrentPath + "\\Choice\\"
F_filepath = CurrentPath + "\\FactorCal\\"

ts.set_token("73dfa0347b3606ad66fb3936d3f0d98ef557a1e538271e811bba3627")
pro = ts.pro_api()

def set_params(file_list):
    new_file_list = []
    for file_name in file_list:
        if file_name[0]=="T":
            new_file_list.append("python " + T_filepath + file_name)
        elif file_name[0]=="C":
            new_file_list.append("python " + C_filepath + file_name)
        else:
            new_file_list.append("python " + F_filepath + file_name)
    return new_file_list

file_list = ['T_SetUp.py']
new_file_list = set_params(file_list)
for pyfile in new_file_list:
    os.system(pyfile)
print("All files update completely!")






