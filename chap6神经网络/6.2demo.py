# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/16 15:20
import tensorflow as tf
from tensorflow.keras import layers

# 6.2.1张量方式实现
# 创建w，b张量
x = tf.random.normal([2, 784])
# 从截断的正态分布中输出随机值。
# 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1  # 批量矩阵相乘函数tf.matmul()完成线性变换
o1 = tf.nn.relu(o1)  # 激活函数
print(o1)


# 6.2.2层方式实现
x = tf.random.normal([4, 28*28])
# 创建全连接层，指定输出节点与激活函数
fc = layers.Dense(512, activation=tf.nn.relu)
h1 = fc(x)  # 通过fc类示例完成一次全连接层的计算，返回输出张量
print(h1)
print(fc.kernel)  # 获取Dense类的权值矩阵
print(fc.bias)  # 获取Dense类的偏置向量
print(fc.trainable_variables)
