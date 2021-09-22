# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 9:32
"""
对于常用的网络模型，如 ResNet、VGG 等，不需要手动创建网络，可以直接从keras.applications 子模块中通过一行代码即可创建并使用这些经典模型，
同时还可以通过设置 weights 参数加载预训练的网络参数，非常方便。

8.5.1加载模型
以 ResNet50 网络模型为例，一般将 ResNet50 去除最后一层后的网络作为新任务的特征提取子网络，
即利用在 ImageNet 数据集上预训练好的网络参数初始化，并根据自定义任务的类别追加一个对应数据类别数的全连接分类层或子网络，
从而可以在预训练网络的基础上快速、高效地学习新任务。

首先利用 Keras 模型乐园加载 ImageNet 预训练好的 ResNet50 网络，代码如下：
# 加载 ImageNet 预训练网络模型，并去掉最后一层
resnet = keras.applications.ResNet50(weights='imagenet',include_top=False)
resnet.summary()
# 测试网络的输出
x = tf.random.normal([4,224,224,3])
out = resnet(x) # 获得子网络的输出
out.shape

上述代码自动从服务器下载模型结构和在 ImageNet 数据集上预训练好的网络参数。通过设置 include_top 参数为 False，可以选择去掉 ResNet50 最后一层，
此时网络的输出特征图大小为[𝑏, 7,7,2048]。对于某个具体的任务，需要设置自定义的输出节点数，以 100 类的分类任务为例，
我们在 ResNet50 基础上重新构建新网络。新建一个池化层(这里的池化层暂时可以理解为高、宽维度下采样的功能)，
将特征从[𝑏, 7,7,2048]降维到[𝑏, 2048]。代码如下：
In [6]:
# 新建池化层
global_average_layer = layers.GlobalAveragePooling2D()
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4,7,7,2048])
# 池化层降维，形状由[4,7,7,2048]变为[4,1,1,2048],删减维度后变为[4,2048]
out = global_average_layer(x)
print(out.shape)
Out[6]: (4, 2048)
最后新建一个全连接层，并设置输出节点数为 100，代码如下：
In [7]:
# 新建全连接层
fc = layers.Dense(100)
# 利用上一层的输出[4,2048]作为本层的输入，测试其输出
x = tf.random.normal([4,2048])
out = fc(x) # 输出层的输出为样本属于 100 类别的概率分布
print(out.shape)
Out[7]: (4, 100)

在创建预训练的 ResNet50 特征子网络、新建的池化层和全连接层后，我们重新利用Sequential 容器封装成一个新的网络：
# 重新包裹成我们的网络模型
mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()
可以看到新的网络模型的结构信息为：
Layer (type) Output Shape Param #
=================================================================
resnet50 (Model) (None, None, None, 2048) 23587712
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048) 0
_________________________________________________________________
dense_4 (Dense) (None, 100) 204900
=================================================================
Total params: 23,792,612
Trainable params: 23,739,492
Non-trainable params: 53,120

通过设置 resnet.trainable = False 可以选择冻结 ResNet 部分的网络参数，只训练新建的网络层，从而快速、高效完成网络模型的训练。
当然也可以在自定义任务上更新网络的全部参数。
"""