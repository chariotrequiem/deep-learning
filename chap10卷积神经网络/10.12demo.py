# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 21:35
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super().__init__()
        # f(x)包含了 2 个普通卷积层，创建卷积层 1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 创建卷积层 2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # 当ℱ(𝒙)的形状与𝒙不同时，无法直接相加，我们需要新建identity(𝒙)卷积层，来完成𝒙的形状转换。紧跟上面代码，实现如下：
        if stride != 1:  # 插入identity层
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:  # 否则，直接连接
            self.downsample = lambda x: x

    # 在前向传播时，只需要将ℱ(𝒙)与identity(𝒙)相加，并添加 ReLU 激活函数即可。前向计算函数代码如下：
    def call(self, inputs, training=None):
        # 前向传播函数
        out = self.conv1(inputs)  # 通过第一个卷积层
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 通过第二个卷积层
        out = self.bn2(out)
        # 输入通过 identity()转换
        identity = self.downsample(inputs)
        # f(x)+x 运算
        output = layers.add([out, identity])
        # 再通过激活函数并返回
        output = tf.nn.relu(output)
        return output

