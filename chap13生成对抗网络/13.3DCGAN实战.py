# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/26 14:47
"""
本节我们来完成一个二次元动漫头像图片生成实战，参考 DCGAN [2]的网络结构，其中判别器 D 利用普通卷积层实现，生成器 G 利用转置卷积层实现，如图 13.6 所示。

13.3.1动漫图片数据集
这里使用的是一组二次元动漫头像的数据集②，共 51223 张图片，无标注信息，图片主 体已裁剪、对齐并统一缩放到96 × 96大小，部分样片如图 13.7 所示
对于自定义的数据集，需要自行完成数据的加载和预处理工作，我们这里聚焦在 GAN算法本身，后续自定义数据集一章会详细介绍如何加载自己的数据集，
这里直接通过预编写好的 make_anime_dataset 函数返回已经处理好的数据集对象。代码如下：
# 数据集路径，从 https://pan.baidu.com/s/1eSifHcA 提取码：g5qa 下载解压
img_path = glob.glob(r'C:\Users\z390\Downloads\faces\*.jpg')
# 构建数据集对象，返回数据集 Dataset 类和图片大小
dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)

其中 dataset 对象就是 tf.data.Dataset 类实例，已经完成了随机打散、预处理和批量化等操作，可以直接迭代获得样本批，img_shape 是预处理后的图片大小。

13.3.2生成器
生成网络 G 由 5 个转置卷积层单元堆叠而成，实现特征图高宽的层层放大，特征图通道数的层层减少。
首先将长度为 100 的隐藏向量𝒛通过 Reshape 操作调整为[𝑏, 1,1,100]的 4维张量，并依序通过转置卷积层，放大高宽维度，减少通道数维度，
最后得到高宽为 64，通道数为 3 的彩色图片。每个卷积层中间插入 BN 层来提高训练稳定性，卷积层选择不使用偏置向量。
生成器的类代码实现如下：
class Generator(keras.Model):
    # 生成器网络类
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        # 转置卷积层 1,输出 channel 为 filter*8,核大小 4,步长 1,不使用 padding,不使用偏置
        self.conv1 = layers.Conv2DTranspose(filter*8, 4,1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 转置卷积层 2
        self.conv2 = layers.Conv2DTranspose(filter*4, 4,2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 转置卷积层 3
        self.conv3 = layers.Conv2DTranspose(filter*2, 4,2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 转置卷积层 4
        self.conv4 = layers.Conv2DTranspose(filter*1, 4,2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 转置卷积层 5
        self.conv5 = layers.Conv2DTranspose(3, 4,2, 'same', use_bias=False)

    生成网络 G 的前向传播过程实现如下：
    def call(self, inputs, training=None):
        x = inputs  # [z, 100]
        # Reshape 成 4D 张量，方便后续转置卷积运算:(b, 1, 1, 100)
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.relu(x) # 激活函数
        # 转置卷积-BN-激活函数:(b, 4, 4, 512)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # 转置卷积-BN-激活函数:(b, 8, 8, 256)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # 转置卷积-BN-激活函数:(b, 16, 16, 128)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # 转置卷积-BN-激活函数:(b, 32, 32, 64)
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # 转置卷积-激活函数:(b, 64, 64, 3)
        x = self.conv5(x)
        x = tf.tanh(x) # 输出 x 范围-1~1,与预处理一致

        return x
生成网络的输出大小为[𝑏, 64,64,3]的图片张量，数值范围为−1~1。

13.3.3判别器
判别网络 D 与普通的分类网络相同，接受大小为[𝑏, 64,64,3]的图片张量，连续通过 5个卷积层实现特征的层层提取，卷积层最终输出大小为[𝑏, 2,2,1024]，
再通过池化层GlobalAveragePooling2D 将特征大小转换为[𝑏, 1024]，最后通过一个全连接层获得二分类任务的概率。
判别网络 D 类的代码实现如下：
class Discriminator(keras.Model):
    # 判别器类
    def __init__(self):
        super(Discriminator, self).__init__()
        filter = 64
        # 卷积层 1
        self.conv1 = layers.Conv2D(filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 卷积层 2
        self.conv2 = layers.Conv2D(filter*2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 卷积层 3
        self.conv3 = layers.Conv2D(filter*4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 卷积层 4
        self.conv4 = layers.Conv2D(filter*8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 卷积层 5
        self.conv5 = layers.Conv2D(filter*16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        # 全局池化层
        self.pool = layers.GlobalAveragePooling2D()
        # 特征打平层
        self.flatten = layers.Flatten()
        # 2 分类全连接层
        self.fc = layers.Dense(1)
判别器 D 的前向计算过程实现如下：
    def call(self, inputs, training=None):
        # 卷积-BN-激活函数:(4, 31, 31, 64)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # 卷积-BN-激活函数:(4, 4, 4, 512)
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        # 卷积-BN-激活函数:(4, 2, 2, 1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        # 卷积-BN-激活函数:(4, 1024)
        x = self.pool(x)
        # 打平
        x = self.flatten(x)
        # 输出，[b, 1024] => [b, 1]
        logits = self.fc(x)

        return logits
判别器的输出大小为[𝑏, 1]，类内部没有使用 Sigmoid 激活函数，通过 Sigmoid 激活函数后可获得𝑏个样本属于真实样本的概率 。


13.3.4训练和可视化
判别网络 根据式(13-1)，判别网络的训练目标是最大化ℒ(𝐷, 𝐺)函数，使得真实样本预测为真的概率接近于 1，生成样本预测为真的概率接近于 0。
我们将判断器的误差函数实现在 d_loss_fn 函数中，将所有真实样本标注为 1，所有生成样本标注为 0，并通过最小化对应的交叉熵损失函数来实现最大化ℒ(𝐷,𝐺)函数。
d_loss_fn 函数实现如下：
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算判别器的误差函数
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)
    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 真实图片与 1 之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与 0 之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss
其中 celoss_ones 函数计算当前预测概率与标签 1 之间的交叉熵损失，代码如下：
def celoss_ones(logits):
    # 计算属于与标签为 1 的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

celoss_zeros 函数计算当前预测概率与标签 0 之间的交叉熵损失，代码如下：
def celoss_zeros(logits):
    # 计算属于与便签为 0 的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

生成网络 的训练目标是最小化ℒ(𝐷, 𝐺)目标函数，由于真实样本与生成器无关，因此误差函数只需要考虑最小化𝔼𝒛~𝑝𝑧(∙)log (1 − 𝐷𝜃(𝐺𝜙(𝒛)))项即可。
可以通过将生成的样本标注为 1，最小化此时的交叉熵误差。需要注意的是，在反向传播误差的过程中，判别器也参与了计算图的构建，
但是此阶段只需要更新生成器网络参数，而不更新判别器的网络参数。
生成器的误差函数代码如下：
def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与 1 之间的误差
    loss = celoss_ones(d_fake_logits)

    return loss

网络训练 在每个 Epoch，首先从先验分布𝑝 (∙)中随机采样隐藏向量，从真实数据集中随机采样真实图片，通过生成器和判别器计算判别器网络的损失，
并优化判别器网络参数𝜃。在训练生成器时，需要借助于判别器来计算误差，但是只计算生成器的梯度信息并更新𝜙。这里设定判别器训练𝑘 = 5次后，生成器训练一次。

首先创建生成网络和判别网络，并分别创建对应的优化器。
代码如下：
generator = Generator() # 创建生成器
generator.build(input_shape = (4, z_dim))
discriminator = Discriminator() # 创建判别器
discriminator.build(input_shape=(4, 64, 64, 3))
# 分别为生成器和判别器创建优化器
g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1= 0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

主训练部分代码实现如下：
for epoch in range(epochs): # 训练 epochs 次
    # 1. 训练判别器
    for _ in range(5):
        # 采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = next(db_iter) # 采样真实图片
        # 判别器前向计算
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    # 2. 训练生成器
    # 采样隐藏向量
    batch_z = tf.random.normal([batch_size, z_dim])
    batch_x = next(db_iter) # 采样真实图片
    # 生成器前向计算
    with tf.GradientTape() as tape:
        g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

每间隔 100 个 Epoch，进行一次图片生成测试。通过从先验分布中随机采样隐向量，送入生成器获得生成图片，并保存为文件。
如图 13.8 所示，展示了 DCGAN 模型在训练过程中保存的生成图片样例，可以观察到，大部分图片主体明确，色彩逼真，图片多样性较丰富，
图片效果较为贴近数据集中真实的图片。同时也能发现仍有少量生成图片损坏，无法通过人眼辨识图片主体。


"""