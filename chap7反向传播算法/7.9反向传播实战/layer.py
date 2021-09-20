# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/20 15:37
import numpy as np


class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param int n_input: 输入节点数
        :param int n_neurons: 输出节点数
        :param str activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """
        # np.random.randn  从标准正态分布中返回一个或多个样本值
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        # numpy.random.rand(d0, d1, …, dn)的从[0, 1)区间中随机返回一个值。
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation  # 激活函数类型，如'Sigmoid'
        self.last_activation = None  # 激活函数的输出值o
        self.error = None  # 用于计算当前层的delta变量的中间变量
        self.delta = None  # 记录当前层的delta变量，用于计算梯度

    # 网络层的前向传播函数实现如下，其中last_activation变量用于保存当前层的输出值
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    # 上面代码中的self._apply_activation 函数实现了不同类型的激活函数的前向计算过程
    # 尽管此处我们只使用Sigmoid激活函数一种。代码如下：
    def _apply_activation(self, r):
        # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数，直接返回
        # ReLU 激活函数
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        # tanh 激活函数
        elif self.activation == 'tanh':
            return np.tanh(r)
        # sigmoid 激活函数
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    # 针对不同类型的激活函数，他们的导数计算实现如下
    def apply_activation_derivative(self, r):
        # 计算激活函数的导数
        # 无激活函数，导数为1
        if self.activation is None:
            return np.ones_like(r)
        # ReLU函数的导数实现
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        # tanh函数的导数实现
        elif self.activation == 'tanh':
            return 1 - r ** 2
        # Sigmoid 函数的导数实现
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r