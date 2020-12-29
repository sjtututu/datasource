#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/27 15:51
# @Author: Tu Xinglong
# @File  : LoggingPlus.py

import logging
# from logging import handlers
from concurrent_log_handler import ConcurrentRotatingFileHandler


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(
            self,
            filename,
            level='info',
            when='D',
            backCount=10,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
            # fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: \n%(message)s')
        self.logger = logging.getLogger(filename)  # 创建一个Logger日志对象
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        # ==================================================
        # ==================================================
        # 防止出现日志出现重复记录和打印问题
        if not self.logger.handlers:
            format_str = logging.Formatter(fmt)  # 设置日志格式
            sh = logging.StreamHandler()  # 设置屏幕输出对象
            sh.setFormatter(format_str)  # 设置屏幕上显示的格式
            # 在多进程中添加锁机制保护文件按照顺序写入
            th = ConcurrentRotatingFileHandler(filename=filename, mode="a", maxBytes=512*1024,
                                               backupCount=backCount, encoding='utf-8')
            # th = handlers.TimedRotatingFileHandler(
            #     filename=filename,
            #     when=when,
            #     backupCount=backCount,
            #     encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
            # 实例化TimedRotatingFileHandler
            # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
            # S 秒
            # M 分
            # H 小时、
            # D 天、
            # W 每星期（interval==0时代表星期一）
            # midnight 每天凌晨
            th.setFormatter(format_str)  # 设置文件里写入的格式
            self.logger.addHandler(sh)  # 把对象加到logger里
            self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('all.log', level='debug')
    import pandas as pd
    import numpy as np
    a = pd.DataFrame([[1,2,3,4],[4,5,6,7]]) 
    b = {"a":12,"b":23,
         "c":53,"d":46}
    c =np.array([[[1,2,3],[1,2,3],[1,2,4]]])
    log.logger.debug(c)
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error.log', level='error').logger.error('error')
    # 但是当发生异常时，直接使用无参数的 debug()、info()、warning()、
    # error()、critical() 方法并不能记录异常信息，需要设置 exc_info
    # 参数为 True 才可以，或者使用 exception() 方法，还可以使用 log()
    # 方法，但还要设置日志级别和 exc_info 参数。
    # try:
    #     c = a / b
    # except Exception as e:
    #     # 下面三种方式三选一，推荐使用第一种
    #     logging.exception("Exception occurred")
    #     logging.error("Exception occurred", exc_info=True)
    #     logging.log(level=logging.DEBUG, msg="Exception occurred", exc_info=True)
