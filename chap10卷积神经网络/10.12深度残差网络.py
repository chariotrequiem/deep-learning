# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 21:11
"""
AlexNet、VGG、GoogLeNet 等网络模型的出现将神经网络的发展带入了几十层的阶段，研究人员发现网络的层数越深，越有可能获得更好的泛化能力。
但是当模型加深以后，网络变得越来越难训练，这主要是由于梯度弥散和梯度爆炸现象造成的。在较深层数的神经网络中，
梯度信息由网络的末层逐层传向网络的首层时，传递的过程中会出现梯度接近于 0 或梯度值非常大的现象。网络层数越深，这种现象可能会越严重。

那么怎么解决深层神经网络的梯度弥散和梯度爆炸现象呢？一个很自然的想法是，既然浅层神经网络不容易出现这些梯度现象，
那么可以尝试给深层神经网络添加一种回退到浅层神经网络的机制。当深层神经网络可以轻松地回退到浅层神经网络时，
深层神经网络可以获得与浅层神经网络相当的模型性能，而不至于更糟糕。

通过在输入和输出之间添加一条直接连接的 Skip Connection 可以让神经网络具有回退的能力。以 VGG13 深度神经网络为例，
假设观察到 VGG13 模型出现梯度弥散现象，而10 层的网络模型并没有观测到梯度弥散现象，那么可以考虑在最后的两个卷积层添加 Skip Connection，
如图 10.62 中所示。通过这种方式，网络模型可以自动选择是否经由这两个卷积层完成特征变换，还是直接跳过这两个卷积层而选择 Skip Connection，
亦或结合两个卷积层和 Skip Connection 的输出。

2015 年，微软亚洲研究院何凯明等人发表了基于 Skip Connection 的深度残差网络(Residual Neural Network，简称 ResNet)算法 [10]，
并提出了 18 层、34 层、50 层、101层、152 层的 ResNet-18、ResNet-34、ResNet-50、ResNet-101 和 ResNet-152 等模型，
甚至成功训练出层数达到 1202 层的极深层神经网络。ResNet 在 ILSVRC 2015 挑战赛 ImageNet数据集上的分类、检测等任务上面均获得了最好性能，
ResNet 论文至今已经获得超 25000的引用量，可见 ResNet 在人工智能行业的影响力。

10.12.1ResNet原理
ResNet 通过在卷积层的输入和输出之间添加 Skip Connection 实现层数回退机制，如下图 10.63 所示，输入𝒙通过两个卷积层，
得到特征变换后的输出ℱ(𝒙)，与输入𝒙进行对应元素的相加运算，得到最终输出
                                ℋ(𝒙)：ℋ(𝒙) = 𝒙 + ℱ(𝒙)
ℋ(𝒙)叫作残差模块(Residual Block，简称 ResBlock)。由于被 Skip Connection 包围的卷积神经网络需要学习映射ℱ(𝒙) = ℋ(𝒙) − 𝒙，
故称为残差网络。

为了能够满足输入𝒙与卷积层的输出ℱ(𝒙)能够相加运算，需要输入𝒙的 shape 与ℱ(𝒙)的shape 完全一致。当出现 shape 不一致时，
一般通过在 Skip Connection 上添加额外的卷积运算环节将输入𝒙变换到与ℱ(𝒙)相同的 shape，如图 10.63 中identity(𝒙)函数所示，
其中identity(𝒙)以 × 的卷积运算居多，主要用于调整输入的通道数。

下图 10.64 对比了 34 层的深度残差网络、34 层的普通深度网络以及 19 层的 VGG 网络结构。可以看到，深度残差网络通过堆叠残差模块，
达到了较深的网络层数，从而获得了训练稳定、性能优越的深层网络模型。

10.12.2ResBlock实现
深度残差网络并没有增加新的网络层类型，只是通过在输入和输出之间添加一条 Skip Connection，因此并没有针对 ResNet 的底层实现。
在 TensorFlow 中通过调用普通卷积层即可实现残差模块。

首先创建一个新类，在初始化阶段创建残差块中需要的卷积层、激活函数层等，首先新建ℱ(𝑥)卷积层，
代码如下：
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
"""