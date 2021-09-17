# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 15:45
import tensorflow as tf
from tensorflow import keras

# 均方差误差函数(MSE)
o = tf.random.normal([2, 10])  # 构造网络输出
y_onehot = tf.constant([1, 3])  # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = keras.losses.MSE(y_onehot, o)  # 计算均方差(每个样本的均方差)
print(loss)
loss = tf.reduce_mean(loss)  # 计算batch均方差
print(loss)

# 使用层方式实现计算均方差
criteon = keras.losses.MeanSquaredError()
loss = criteon(y_onehot, o)  # 计算batch均方差
print(loss)