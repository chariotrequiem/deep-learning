# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 21:16
from tensorflow import keras
from tensorflow.keras import layers


class Network(keras.Model):
    # 回归网络模型
    def __init__(self):
        super(Network, self).__init__()
        # 创建3个全连接层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # 依次通过3个全连接层
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)

        return x