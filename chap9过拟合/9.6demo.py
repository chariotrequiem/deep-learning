# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 14:32
import tensorflow as tf
from tensorflow.keras import models

# 添加dropout操作，断开概率为0.5
x = tf.nn.dropout(x, rate=0.5)
# 添加dropout层，断开概率为0.5
model.add(layers.Dropout(rate=0.5))
