# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 19:21
import tensorflow as tf
from tensorflow.keras import layers


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

x = tf.random.normal([1, 7, 7, 1])  # 模拟输入
# 空洞卷积，1 个 3x3 的卷积核
layer = layers.Conv2D(1, kernel_size=3, strides=1, dilation_rate=2)
out = layer(x)  # 前向计算
print(out.shape)  # [1, 3, 3, 1]

print('----------------------实现转置卷积运算-------------------------------------')
# 创建X矩阵，高宽为5 X 5
x = tf.range(25) + 1
# Reshape为合法维度的张量
x = tf.reshape(x, [1, 5, 5, 1])
x = tf.cast(x, tf.float32)
# 创建固定内容的卷积核矩阵
w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])
# 调整为合法维度的张量
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
print(w.shape)
# 进行普通卷积运算
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
# print(out.shape)  # (1, 2, 2, 1)


# 普通卷积的输出作为转置卷积的输入，进行转置卷积运算
xx = tf.nn.conv2d_transpose(out, w, strides=2, padding='VALID', output_shape=[1, 5, 5, 1])
print(xx)


print('-------------------𝒐 + 𝟐𝒑 − 𝒌不为𝒔倍数--------------------')
x = tf.random.normal([1, 6, 6, 1])
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
print(out.shape)  # (1, 2, 2, 1)


xx = tf.nn.conv2d_transpose(out, w, strides=2, padding='VALID', output_shape=[1, 6, 6, 1])
print(xx)  # shape=(1, 6, 6, 1)

