# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 17:15
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, losses, models


"""# 重新创建相同的网络结构
network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
# 从参数文件中读取数据并写入当前网络
network.load_weights('weights.ckpt')
print('loaded weights!')"""


# 首先创建5层的全连接神经网络
network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(4, 28*28))
print(network.summary())

# 模型装配
network.compile(optimizer=optimizers.Adam(lr=0.01),
                # from_logits: True就是需要经过softmax进行概率化，默认False及y_pred是经过softmax处理的
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

"""network.save('model.h5')
print('saved total model.')
del network

network = models.load_model('model.h5'.encode('utf-8').decode('utf-8'))"""


"""tf.saved_model.save(network, 'model-savedmodel')
print('saving savemodel')

del network
print('load savedmodel from file.') # 从文件恢复网络结构与网络参数
network = tf.saved_model.load('model-savedmodel') # 准确率计量器
acc_meter = metrics.CategoricalAccuracy()
for x,y in ds_val: # 遍历测试集
pred = network(x) # 前向计算
acc_meter.update_state(y_true=y, y_pred=pred) # 更新准确率统计
# 打印准确率
print("Test Accuracy:%f" % acc_meter.result())"""
