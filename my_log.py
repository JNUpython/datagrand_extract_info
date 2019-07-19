# -*- coding: utf-8 -*-
# @Time    : 2019/7/6 12:54
# @Author  : kean
# @Email   : ?
# @File    : mylogger.py
# @Software: PyCharm


import logging

"""通过handler控制输出console以及log file的输出格式
format参数中可能用到的格式化串：
%(name)s             Logger的名字
%(levelno)s          数字形式的日志级别
%(levelname)s     文本形式的日志级别
%(pathname)s     调用日志输出函数的模块的完整路径名，可能没有
%(filename)s        调用日志输出函数的模块的文件名
%(module)s          调用日志输出函数的模块名
%(funcName)s     调用日志输出函数的函数名
%(lineno)d           调用日志输出函数的语句所在的代码行
%(created)f          当前时间，用UNIX标准的表示时间的浮 点数表示
%(relativeCreated)d    输出日志信息时的，自Logger创建以 来的毫秒数
%(asctime)s                字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”。逗号后面的是毫秒
%(thread)d                 线程ID。可能没有
%(threadName)s        线程名。可能没有
%(process)d              进程ID。可能没有
%(message)s            用户输出的消息
"""
# logging.basicConfig(level=logging.INFO)

BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(filename)s(%(lineno)d) %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
# fhlr = logging.FileHandler('example.log')  # 输出到文件的handler
# fhlr.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel('DEBUG')
logger.addHandler(chlr)


# logger.addHandler(fhlr)

# logger.info('this is info')
# logger.debug('this is debug')
# logger.info("test logging format")


def print_falgs(FLAGS):
    print('=================My Configurations:=====================')
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value = value.value
        if type(value) == float:
            print(' %s:\t %f' % (name, value))
        elif type(value) == int:
            print(' %s:\t %d' % (name, value))
        elif type(value) == str:
            print(' %s:\t %s' % (name, value))
        elif type(value) == bool:
            print(' %s:\t %s' % (name, value))
        else:
            print('%s:\t %s' % (name, value))
    # for k, v in sorted(FLAGS.__dict__.items()):
    # print(f'{k}={v}\n')
    print('=================My Configurations:=====================')
