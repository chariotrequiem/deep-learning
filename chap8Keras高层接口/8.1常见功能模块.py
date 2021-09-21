# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/20 16:56
"""
Keras 提供了一系列高层的神经网络相关类和函数，如经典数据集加载函数、网络层类、模型容器、损失函数类、优化器类、经典模型类等。
对于经典数据集，通过一行代码即可下载、管理、加载数据集，这些数据集包括Boston 房价预测数据集、CIFAR 图片数据集、
MNIST/FashionMNIST 手写数字图片数据集、IMDB 文本数据集等。我们已经介绍过，不再敖述。

8.1.1常见网络层类
对于常见的神经网络层，可以使用张量方式的底层接口函数来实现，这些接口函数一般在 tf.nn 模块中。
更常用地，对于常见的网络层，我们一般直接使用层方式来完成模型的搭建，
在 tf.keras.layers 命名空间(下文使用 layers 指代 tf.keras.layers)中提供了大量常见网络层的类，
如全连接层、激活函数层、池化层、卷积层、循环神经网络层等。对于这些网络层类，只需要在创建时指定网络层的相关参数，
并调用__call__方法即可完成前向计算。在调用__call__方法时，Keras 会自动调用每个层的前向传播逻辑，这些逻辑一般实现在类的call 函数中。

以 Softmax 层为例，它既可以使用 tf.nn.softmax 函数在前向传播逻辑中完成 Softmax运算，
也可以通过 layers.Softmax(axis)类搭建 Softmax 网络层，其中 axis 参数指定进行softmax 运算的维度。首先导入相关的子模块，
实现如下：
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers # 导入常见网络层类
然后创建 Softmax 层，并调用__call__方法完成前向计算：
In [1]:
x = tf.constant([2.,1.,0.1]) # 创建输入张量
layer = layers.Softmax(axis=-1) # 创建 Softmax 层
out = layer(x) # 调用 softmax 前向计算，输出为 out
经过 Softmax 网络层后，得到概率分布 out 为：
Out[1]:
<tf.Tensor: id=2, shape=(3,), dtype=float32, numpy=array([0.6590012,
0.242433 , 0.0985659], dtype=float32)>
当然，也可以直接通过 tf.nn.softmax()函数完成计算，代码如下：
out = tf.nn.softmax(x) # 调用 softmax 函数完成前向计算


8.1.2网络容器
对于常见的网络，需要手动调用每一层的类实例完成前向传播运算，当网络层数变得较深时，这一部分代码显得非常臃肿。
可以通过 Keras 提供的网络容器 Sequential 将多个网络层封装成一个大网络模型，
只需要调用网络模型的实例一次即可完成数据从第一层到最末层的顺序传播运算。
例如，2 层的全连接层加上单独的激活函数层，可以通过 Sequential 容器封装为一个网络。
# 导入 Sequential 容器
from tensorflow.keras import layers, Sequential
network = Sequential([ # 封装为一个网络
 layers.Dense(3, activation=None), # 全连接层，此处不使用激活函数
 layers.ReLU(),#激活函数层
 layers.Dense(2, activation=None), # 全连接层，此处不使用激活函数
 layers.ReLU() #激活函数层
])
x = tf.random.normal([4,3])
out = network(x) # 输入从第一层开始，逐层传播至输出层，并返回输出层的输出
Sequential 容器也可以通过 add()方法继续追加新的网络层，实现动态创建网络的功能：
In [2]:
layers_num = 2 # 堆叠 2 次
network = Sequential([]) # 先创建空的网络容器
for _ in range(layers_num):
    network.add(layers.Dense(3)) # 添加全连接层
    network.add(layers.ReLU())# 添加激活函数层
network.build(input_shape=(4, 4)) # 创建网络参数
network.summary()
上述代码通过指定任意的 layers_num 参数即可创建对应层数的网络结构，在完成网络创建时，网络层类并没有创建内部权值张量等成员变量，
此时通过调用类的 build 方法并指定输入大小，即可自动创建所有层的内部张量。通过 summary()函数可以方便打印出网络结构和参数量，
打印结果如下：
Out[2]:
Model: "sequential_2"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
dense_2 (Dense) multiple 15
_________________________________________________________________
re_lu_2 (ReLU) multiple 0
_________________________________________________________________
dense_3 (Dense) multiple 12
_________________________________________________________________
re_lu_3 (ReLU) multiple 0
=================================================================
Total params: 27
Trainable params: 27
Non-trainable params: 0
可以看到 Layer 列为每层的名字，这个名字由 TensorFlow 内部维护，与 Python 的对象名并不一样。Param#列为层的参数个数，
Total params 项统计出了总的参数量，Trainable params为总的待优化参数量，Non-trainable params 为总的不需要优化的参数量。
读者可以简单验证一下参数量的计算结果。

当我们通过 Sequential 容量封装多个网络层时，每层的参数列表将会自动并入Sequential 容器的参数列表中，不需要人为合并网络参数列表，
这也是 Sequential 容器的便捷之处。Sequential 对象的 trainable_variables 和 variables 包含了所有层的待优化张量列表和全部张量列表，
例如：
In [3]: # 打印网络的待优化参数名与 shape
for p in network.trainable_variables:
    print(p.name, p.shape) # 参数名和形状
Out[3]:
dense_2/kernel:0 (4, 3)
dense_2/bias:0 (3,)
dense_3/kernel:0 (3, 3)
dense_3/bias:0 (3,)
Sequential 容器是最常用的类之一，对于快速搭建多层神经网络非常有用，应尽量多使用来简化网络模型的实现。

"""