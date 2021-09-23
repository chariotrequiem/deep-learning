# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 10:42
import tensorflow as tf

# 构造输入
x = tf.random.normal([100, 32, 32, 3])
# 将其他维度合并，仅保留通道维度
x = tf.reshape(x, [-1, 3])
print(x.shape)
# 计算其他维度的均值
ub = tf.reduce_mean(x, axis=0)
print(ub)