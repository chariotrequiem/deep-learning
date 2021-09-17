# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 14:32
import tensorflow as tf

# 使用Sigmoid函数将向量中的元素值由[-6, 6]映射到(0, 1)
x = tf.linspace(-6., 6., 10)
print(x)
x = tf.nn.sigmoid(x)
print(x)


# 实现ReLU()函数
x = tf.random.normal([10])
print(x)
y = tf.nn.relu(x)
print(y)


# 实现Leaky_ReLU函数
x = tf.linspace(-6., 6., 10)
print(x)
y = tf.nn.leaky_relu(x, alpha=0.1)
print(y)


# 实现tanh函数
x = tf.linspace(-10., 10., 10)
print(x)
y = tf.nn.tanh(x)  # 将输入"压缩"到(-1, 1)之间
print(y)