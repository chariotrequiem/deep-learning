# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 10:56
import tensorflow as tf
"""
x = tf.random.normal([10, 35, 8])
result = tf.split(x, num_or_size_splits=10, axis=0)
print(len(result))
print(result[0])

y = tf.random.normal([10, 35, 8])
result = tf.split(y, num_or_size_splits=[4, 2, 2, 2], axis=0)
len(result)
print(result[0])

z = tf.random.normal([10, 35, 8])
result = tf.unstack(z, axis=0)
print(len(result))"""

"""# 向量范数
import numpy as np
x = tf.ones([2, 2])
print(tf.norm(x, ord=1))
print(tf.norm(x, ord=2))
print(tf.norm(x, ord=np.inf))"""

"""# tf.reduce_max()函数

x = tf.random.normal([4, 10])
print(x)
print(tf.reduce_max(x, axis=1))  # 每个样本的最大概率值
print(tf.reduce_min(x, axis=1))  # 每个样本额最小概率值
print(tf.reduce_mean(x, axis=1))  # 每个样本的概率均值
print(tf.reduce_sum(x, axis=1))  # 张量在axis=1轴上所有的特征的和

from tensorflow import keras
out = tf.random.normal([4, 10])
y = tf.constant([1, 2, 2, 0])
y = tf.one_hot(y, depth=10)
loss = keras.losses.mse(y, out)
print(loss)
loss = tf.reduce_mean(loss)
print(loss)"""


"""# tf.argmax(x, axis)和tf.argmin(x, axis)可以求解在axis轴上，x的最大值、最小值所在的索引号
out = tf.random.normal([2, 10])
print(out)
out = tf.nn.softmax(out, axis=1)  # 通过 softmax 函数转换为概率值
print(out)
print(tf.argmax(out, axis=1))"""

