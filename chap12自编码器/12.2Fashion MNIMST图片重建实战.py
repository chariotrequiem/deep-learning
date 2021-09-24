# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/24 22:06
"""
自编码器算法原理非常简单，实现方便，训练也较稳定，相对于 PCA 算法，神经网络的强大表达能力可以学习到输入的高层抽象的隐藏特征向量𝒛，
同时也能够基于𝒛重建出输入。这里我们基于 Fashsion MNIST 数据集进行图片重建实战。

12.2.1Fashion MNIST数据集
Fashion MNIST 是一个定位在比 MNIST 图片识别问题稍复杂的数据集，它的设定与MNIST 几乎完全一样，包含了 10 类不同类型的衣服、鞋子、包等灰度图片，
图片大小为28 × 28，共 70000 张图片，其中 60000 张用于训练集，10000 张用于测试集，如图 12.4所示，每行是一种类别图片。
可以看到，Fashion MNIST 除了图片内容与 MNIST 不一样，其它设定都相同，大部分情况可以直接替换掉原来基于 MNIST 训练的算法代码，
而不需要额外修改。由于 Fashion MNIST 图片识别相对于 MNIST 图片更难，因此可以用于测试稍复杂的算法性能。

在 TensorFlow 中，加载 Fashion MNIST 数据集同样非常方便，利用keras.datasets.fashion_mnist.load_data()函数即可在线下载、管理和加载。
代码如下：
# 加载 Fashion MNIST 图片数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 归一化
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# 只需要通过图片数据即可构建数据集对象，不需要标签
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
# 构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)


12.2.2编码器
我们利用编码器将输入图片𝒙 ∈ 𝑅784降维到较低维度的隐藏向量：h∈ 𝑅20，并基于隐藏向量h利用解码器重建图片，自编码器模型如图 12.5 所示，
编码器由 3 层全连接层网络组成，输出节点数分别为 256、128、20，解码器同样由 3 层全连接网络组成，输出节点数分别为 128、256、784。

首先是编码器子网络的实现。利用 3 层的神经网络将长度为 784 的图片向量数据依次降维到 256、128，最后降维到 h_dim 维度，
每层使用 ReLU 激活函数，最后一层不使用激活函数。代码如下：

# 创建 Encoders 网络，实现在自编码器类的初始化函数中
self.encoder = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(h_dim)
])


12.2.3解码器
然后再来创建解码器子网络，这里基于隐藏向量 h_dim 依次升维到 128、256、784 长 度，除最后一层，激活函数使用 ReLU 函数。
解码器的输出为 784 长度的向量，代表了打平后的28 × 28大小图片，通过 Reshape 操作即可恢复为图片矩阵。
代码如下：
# 创建 Decoders 网络
self.decoder = Sequential([
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(784)
])


12.2.4自编码器
上述的编码器和解码器 2 个子网络均实现在自编码器类 AE 中，我们在初始化函数中
同时创建这两个子网络。代码如下：
class AE(keras.Model):
    # 自编码器模型类，包含了 Encoder 和 Decoder2 个子网络
    def __init__(self):
        super(AE, self).__init__()
        # 创建 Encoders 网络
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        # 创建 Decoders 网络
            self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])
接下来将前向传播过程实现在 call 函数中，输入图片首先通过 encoder 子网络得到隐藏向量 h，再通过 decoder 得到重建图片。
依次调用编码器和解码器的前向传播函数即可，
代码如下：
    def call(self, inputs, training=None):
        # 前向传播函数
        # 编码获得隐藏向量 h,[b, 784] => [b, 20]
        h = self.encoder(inputs)
        # 解码获得重建图片，[b, 20] => [b, 784]
        x_hat = self.decoder(h)
        return x_hat

12.2.5网络训练
自编码器的训练过程与分类器的基本一致，通过误差函数计算出重建向量𝒙 与原始输入向量𝒙之间的距离，
再利用 TensorFlow 的自动求导机制同时求出 encoder 和 decoder 的梯度，循环更新即可。
首先创建自编码器实例和优化器，并设置合适的学习率。
例如：
# 创建网络对象
model = AE()
# 指定输入大小
model.build(input_shape=(4, 784))
# 打印网络信息
model.summary()
# 创建优化器，并设置学习率
optimizer = optimizers.Adam(lr=lr)

这里固定训练 100 个 Epoch，每次通过前向计算获得重建图片向量，并利用tf.nn.sigmoid_cross_entropy_with_logits 损失函数计算重建图片与原始图片直接的误差，
实际上利用 MSE 误差函数也是可行的。代码如下：
for epoch in range(100): # 训练 100 个 Epoch
    for step, x in enumerate(train_db): # 遍历训练集
        # 打平，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # 构建梯度记录器
        with tf.GradientTape() as tape:
            # 前向计算获得重建的图片
            x_rec_logits = model(x)
            # 计算重建图片与输入之间的损失函数
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            # 计算均值
            rec_loss = tf.reduce_mean(rec_loss)

        # 自动求导，包含了 2 个子网络的梯度
        grads = tape.gradient(rec_loss, model.trainable_variables)
        # 自动更新，同时更新 2 个子网络
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 ==0: # 间隔性打印训练误差
            print(epoch, step, float(rec_loss))

12.2.6图片重建
与分类问题不同的是，自编码器的模型性能一般不好量化评价，尽管ℒ值可以在一定程度上代表网络的学习效果，但我们最终希望获得还原度较高、样式较丰富的重建样本。
因此一般需要根据具体问题来讨论自编码器的学习效果，比如对于图片重建，一般依赖于人工主观评价图片生成的质量，
或利用某些图片逼真度计算方法(如 Inception Score 和 Frechet Inception Distance)来辅助评估。

为了测试图片重建效果，我们把数据集切分为训练集与测试集，其中测试集不参与训练。我们从测试集中随机采样测试图片𝒙 ∈ 𝔻test，
经过自编码器计算得到重建后的图片，然后将真实图片与重建图片保存为图片阵列，并可视化，方便比对。
代码如下：
        # 重建图片，从测试集采样一批图片
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784])) # 打平并送入自编码器
        x_hat = tf.sigmoid(logits) # 将输出转换为像素值，使用 sigmoid 函数
        # 恢复为 28x28,[b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        # 输入的前 50 张+重建的前 50 张图片合并，[b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
        x_concat = x_concat.numpy() * 255. # 恢复为 0~255 范围
        x_concat = x_concat.astype(np.uint8) # 转换为整型
        save_images(x_concat, 'ae_images/rec_epoch_%d.png'%epoch) # 保存图片

图片重建的效果如图 12.6、图 12.7、图 12.8 所示，其中每张图片的左边 5 列为真实图片，右边 5 列为对应的重建图片。
可以看到，第一个 Epoch 时，图片重建效果较差，图片非常模糊，逼真度较差；随着训练的进行，重建图片边缘越来越清晰，
第 100 个 Epoch时，重建的图片效果已经比较接近真实图片。

这里的 save_images 函数负责将多张图片合并并保存为一张大图，这部分代码使用 PIL图片库完成图片阵列逻辑，
代码如下：
def save_images(imgs, name):
    # 创建 280x280 大小图片阵列
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28): # 10 行图片阵列
        for j in range(0, 280, 28): # 10 列图片阵列
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j)) # 写入对应位置
            index += 1 # 保存图片阵列
    new_im.save(name)
"""