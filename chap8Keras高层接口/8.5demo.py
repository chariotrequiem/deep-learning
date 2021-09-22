# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 9:39
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# 加载ImageNet预训练网络模型，并去掉最后一层
resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)
resnet.summary()
# 测试网络的输出
x = tf.random.normal([4, 224, 224, 3])
out = resnet(x)
print(out.shape)


# 新建池化层
global_average_layer = layers.GlobalAveragePooling2D()
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4, 7, 7, 2048])
# 池化层降维，形状由[4, 7, 7, 2048]变为[4, 1, 1, 2048],删减维度后变为[4, 2048]
out = global_average_layer(x)
print(out.shape)

# 新建一个全连接层，设置输出节点为100
# 新建全连接层
fc = layers.Dense(100)
# 利用上一层的输出[4， 2048]作为样本层的输入，测试其输出
x = tf.random.normal([4, 2048])
out = fc(x)  # 输出层的输出为样本属于100类别的概率分布
print(out.shpae)

mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()