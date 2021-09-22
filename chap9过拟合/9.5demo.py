# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 14:20
import tensorflow as tf

# 创建网络参数w1，w2
w1 = tf.random.normal([4, 3])
w2 = tf.random.normal([4, 2])
# 计算L1正则化项
loss_reg = tf.reduce_sum(tf.math.abs(w1)) + tf.reduce_sum(tf.math.abs(w2))
print(loss_reg)

loss_reg = tf.reduce_sum(tf.square(w1)) + tf.reduce_sum(tf.square(w2))
print(loss_reg)