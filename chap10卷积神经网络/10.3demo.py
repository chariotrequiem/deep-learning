# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 20:47
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

x = tf.random.normal([2, 5, 5, 3])  # 模拟输入，3通道，高宽为5
# 需要根据[k,k,cin,cout]格式创建w张量，4个3*3大小卷积核
w = tf.random.normal([3, 3, 3, 4])
# 步长为1，padding为0
out = tf.nn.conv2d(x, w, strides=1, padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
# 输出张量的shape
print(out.shape)


out = tf.nn.conv2d(x, w, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
print(out.shape)

# 通过设置参数padding='SAME'、strides=1可以直接得到输入、输出同大小的卷积层
out = tf.nn.conv2d(x, w, strides=1, padding='SAME')
print(out.shape)


layer = layers.Conv2D(4, kernel_size=3, strides=1, padding='SAME')
out = layer(x)
print(out.shape)
print(layer.trainable_variables)