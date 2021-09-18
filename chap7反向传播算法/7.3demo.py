# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 10:21
import tensorflow as tf
import numpy as np


def sigmoid(x):  # 实现sigmoid函数
    return 1 / (1 + np.exp(-x))


def derivative(x):  # sigmoid导数的计算
    return sigmoid(x) * (1 - sigmoid(x))


def derivative_relu(x):  # ReLU函数的的导数
    d = np.array(x, copy=True)  # 用于保存梯度的张量
    d[x < 0] = 0  # 元素为负的导数为0
    d[x > 0] = 1  # 元素为正的导数为1
    return d


def derivative_l(x, p):  # 其中 p 为 LeakyReLU 的负半段斜率，为超参数
    dx = np.ones_like(x)  # 创建梯度张量，全部初始化为 1
    dx[x < 0] = p  # 元素为负的导数为 p
    return dx


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def derivative_t(x):  # tanh导数实现
    return 1 - tanh(x) ** 2