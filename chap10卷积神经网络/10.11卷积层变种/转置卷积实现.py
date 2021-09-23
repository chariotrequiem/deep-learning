# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 20:35
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


x = tf.range(16) + 1
x = tf.reshape(x, [1, 4, 4, 1])
x = tf.cast(x, tf.float32)
# 创建3*3卷积核
w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
# 普通卷积运算
out = tf.nn.conv2d(x, w, strides=1, padding='VALID')
print(out)  # shape=(1, 2, 2, 1)

"""
在保持 strides=1，padding=’VALID’，卷积核不变的情况下，我们通过卷积核 w 与输出 out 的转置卷积运算尝试恢复与输入 x 相同大小的高宽张量，
代码如下：
"""
xx = tf.nn.conv2d_transpose(out, w, strides=1, padding='VALID', output_shape=[1, 4, 4, 1])
print(xx)


"""
转置卷积也可以和其他层一样，通过 layers.Conv2DTranspose 类创建一个转置卷积层，然后调用实例即可完成前向计算
"""
layer = layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='VALID')
xx2 = layer(out)
print(xx2)