# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 17:04
"""
模型训练完成后，需要将模型保存到文件系统上，从而方便后续的模型测试与部署工作。实际上，在训练时间隔性地保存模型状态也是非常好的习惯，
这一点对于训练大规模的网络尤其重要。一般大规模的网络需要训练数天乃至数周的时长，一旦训练过程被中断或者发生宕机等意外，
之前训练的进度将全部丢失。如果能够间断地保存模型状态到文件系统，即使发生宕机等意外，也可以从最近一次的网络状态文件中恢复，
从而避免浪费大量的训练时间和计算资源。因此模型的保存与加载非常重要。
在 Keras 中，有三种常用的模型保存与加载方法。

8.3.1张量方式
网络的状态主要体现在网络的结构以及网络层内部张量数据上，因此在拥有网络结构源文件的条件下，
直接保存网络张量参数到文件系统上是最轻量级的一种方式。我们以MNIST 手写数字图片识别模型为例，
通过调用 Model.save_weights(path)方法即可将当前的网络参数保存到 path 文件上，代码如下：

network.save_weights('weights.ckpt') # 保存模型的所有张量数据
上述代码将 network 模型保存到 weights.ckpt 文件上。在需要的时候，先创建好网络对象，
然后调用网络对象的 load_weights(path)方法即可将指定的模型文件中保存的张量数值写入到当前网络参数中去，
例如：
# 保存模型参数到文件上
network.save_weights('weights.ckpt')
print('saved weights.')
del network # 删除网络对象
# 重新创建相同的网络结构
network = Sequential([layers.Dense(256, activation='relu'),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),loss=tf.losses.CategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
# 从参数文件中读取数据并写入当前网络
network.load_weights('weights.ckpt')
print('loaded weights!')

这种保存与加载网络的方式最为轻量级，文件中保存的仅仅是张量参数的数值，并没有其它额外的结构参数。
但是它需要使用相同的网络结构才能够正确恢复网络状态，因此一般在拥有网络源文件的情况下使用。



8.3.2网络方式
我们来介绍一种不需要网络源文件，仅仅需要模型参数文件即可恢复出网络模型的方法。
通过 Model.save(path)函数可以将模型的结构以及模型的参数保存到 path 文件上，
在不需要网络源文件的条件下，通过 keras.models.load_model(path)即可恢复网络结构和网络参数。
首先将 MNIST 手写数字图片识别模型保存到文件上，并且删除网络对象：

# 保存模型结构与模型参数到文件
network.save('model.h5')
print('saved total model.')
del network # 删除网络对象
此时通过 model.h5 文件即可恢复出网络的结构和状态，不需要提前创建网络对象，代码如下：
# 从文件恢复网络结构与网络参数
network = keras.models.load_model('model.h5')
可以看到，model.h5 文件除了保存了模型参数外，还应保存了网络结构信息，不需要提前创建模型即可直接从文件中恢复出网络 network 对象。


8.3.3SaveModel方式
TensorFlow 之所以能够被业界青睐，除了优秀的神经网络层 API 支持之外，还得益于它强大的生态系统，包括移动端和网页端等的支持。
当需要将模型部署到其他平台时，采用 TensorFlow 提出的 SavedModel 方式更具有平台无关性。

通过 tf.saved_model.save (network, path)即可将模型以 SavedModel 方式保存到 path 目录中，
代码如下：
# 保存模型结构与模型参数到文件
tf.saved_model.save(network, 'model-savedmodel')
print('saving savedmodel.')
del network # 删除网络对象
此时在文件系统 model-savedmodel 目录上出现了如下网络文件，如图 8.1 所示：
用户无需关心文件的保存格式，只需要通过 tf.saved_model.load 函数即可恢复出模型
对象，我们在恢复出模型实例后，完成测试准确率的计算，实现如下：
print('load savedmodel from file.') # 从文件恢复网络结构与网络参数
network = tf.saved_model.load('model-savedmodel') # 准确率计量器
acc_meter = metrics.CategoricalAccuracy()
for x,y in ds_val: # 遍历测试集
pred = network(x) # 前向计算
acc_meter.update_state(y_true=y, y_pred=pred) # 更新准确率统计
# 打印准确率
print("Test Accuracy:%f" % acc_meter.result())
"""