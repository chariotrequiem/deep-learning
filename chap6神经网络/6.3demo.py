# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/16 16:23
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 6.3.1张量方式实现
# 隐藏层1张量
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隐藏层2张量
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隐藏层3张量
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

x = tf.random.normal([4, 28*28])
# 计算
with tf.GradientTape() as tape:  # 梯度记录器
    # x: [b, 28*28]
    # 隐藏层 1 前向计算，[b, 28*28] => [b, 256]
    h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    # 隐藏层 2 前向计算，[b, 256] => [b, 128]
    h2 = h1 @ w2 + b2
    h2 = tf.nn.relu(h2)
    # 隐藏层 3 前向计算，[b, 128] => [b, 64]
    h3 = h2 @ w3 + b3
    h3 = tf.nn.relu(h3)
    # 输出层前向计算，[b, 64] => [b, 10]
    h4 = h3 @ w4 + b4


# 6.3.2层方式实现
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # 创建隐藏层1
    layers.Dense(128, activation=tf.nn.relu),  # 创建隐藏层2
    layers.Dense(64, activation=tf.nn.relu),  # 创建隐藏层3
    layers.Dense(10, activation=None)  # 创建输出层
])
out = model(x)
print(out)
