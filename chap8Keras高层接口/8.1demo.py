# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 15:31
import tensorflow as tf
# 导入keras模型，不能使用import keras，他导入的是标准的Keras库
from tensorflow import keras
# 导入Sequential容器
from tensorflow.keras import layers, Sequential


x = tf.constant([2., 1., 0.1])  # 创建输入张量
layer = layers.Softmax(axis=-1)  # 创建Softmax层
out = layer(x)  # 调用softmax前向计算，输出为out
print(out)   # 经过Softmax网络层后，得到概率分布out为

# 也可以通过tf.nn.softmax()函数完成计算
out = tf.nn.softmax(x)
print(out)


# 网络容器
network = Sequential([  # 封装为一个网络
    layers.Dense(3, activation=None),  # 全连接层，此处不使用激活函数
    layers.ReLU(),  # 激活函数层
    layers.Dense(2, activation=None),  # 全连接层，此处不适用激活函数
    layers.ReLU()  # 激活函数层
])
x = tf.random.normal([4,3])
# print(x)
out = network(x)  # 输出从第一层开始，逐层传播至输出层，并返回输出层的输出
print(out)

# Sequential容器也可以通过add()放啊继续追加新的网络层，实现动态创建网络的功能：
layers_num = 2  # 堆叠两次
network = Sequential([])  # 先创建空的网络容器
for _ in range(layers_num):
    network.add(layers.Dense(3))  # 添加全连接层
    network.add(layers.ReLU())  # 添加激活函数层
network.build(input_shape=(4, 3))  # 创建网络参数
print(network.summary())


# 打印网络的待优化参数名与shape
for p in network.trainable_variables:
    print(p.name, p.shape)  # 参数名和形状
