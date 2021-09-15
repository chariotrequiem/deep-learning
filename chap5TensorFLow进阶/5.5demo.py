# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 16:45

import tensorflow as tf


def relu(x):
    return tf.maximum(x, 0)


X = tf.random.normal([2, 2])
print(relu(X))


x = tf.range(10)
print(tf.clip_by_value(x, 2, 7))