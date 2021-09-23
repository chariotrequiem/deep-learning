# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 15:47
"""
MNIST 是机器学习最常用的数据集之一，但由于手写数字图片非常简单，并且MNIST 数据集只保存了图片灰度信息，并不适合输入设计为 RGB 三通道的网络模型。
本节将介绍另一个经典的图片分类数据集：CIFAR10。

CIFAR10 数据集由加拿大 Canadian Institute For Advanced Research 发布，它包含了飞机、汽车、鸟、猫等共 10 大类物体的彩色图片，
每个种类收集了 6000 张32 × 32大小图片，共 6 万张图片。其中 5 万张作为训练数据集，1 万张作为测试数据集。每个种类样片如图 10.49 所示。

在 TensorFlow 中，同样地，不需要手动下载、解析和加载 CIFAR10 数据集，通过datasets.cifar10.load_data()函数就可以直接加载切割好的训练集和测试集。
例如：
# 在线下载，加载 CIFAR10 数据集
(x,y), (x_test, y_test) = datasets.cifar10.load_data()
# 删除 y 的一个维度，[b,1] => [b]
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1) # 打印训练集和测试集的形状
print(x.shape, y.shape, x_test.shape, y_test.shape)

# 构建训练集对象，随机打乱，预处理，批量化
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)
# 构建测试集对象，预处理，批量化
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(128)

# 从训练集中采样一个 Batch，并观察
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

TensorFlow 会自动将数据集下载在 C:\Users\用户名\.keras\datasets 路径下，用户可以查看，也可手动删除不需要的数据集缓存。
上述代码运行后，得到训练集的𝑿和𝒚形状为：(50000, 32, 32, 3)和(50000)，测试集的𝑿和𝒚形状为(10000, 32, 32, 3)和(10000)，
分别代表了图片大小为32 × 32，彩色图片，训练集样本数为 50000，测试集样本数为 10000。

CIFAR10 图片识别任务并不简单，这主要是由于 CIFAR10 的图片内容需要大量细节才能呈现，而保存的图片分辨率仅有32 × 32，
使得部分主体信息较为模糊，甚至人眼都很难分辨。浅层的神经网络表达能力有限，很难训练优化到较好的性能，
本节将基于表达能力更强的 VGG13 网络，根据我们的数据集特点修改部分网络结构，完成 CIFAR10 图片识别。
修改如下：

❑ 将网络输入调整为32× 32。原网络输入为22 × 22 ，导致全连接层输入特征维度过大，网络参数量过大
❑ 3 个全连接层的维度调整为 [256, 64, 10] ，满足 10 分类任务的设定。
  图 10.50 是调整后的 VGG13 网络结构，我们统称之为 VGG13 网络模型

我们将网络实现为 2 个子网络：卷积子网络和全连接子网络。卷积子网络由 5 个子模块构成，每个子模块包含了 Conv-Conv-MaxPooling 单元结构，
代码如下：
conv_layers = [ # 先创建包含多网络层的列表
        # Conv-Conv-Pooling 单元 1 # 64 个 3x3 卷积核, 输入输出同大小
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        # 高宽减半
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 2,输出通道提升至 128，高宽大小减半
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 3,输出通道提升至 256，高宽大小减半
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 4,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 5,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        ]

# 利用前面创建的层列表构建网络容器
conv_net = Sequential(conv_layers)

全连接子网络包含了 3 个全连接层，每层添加 ReLU 非线性激活函数，最后一层除外。
代码如下：
# 创建 3 层全连接层子网络
fc_net = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])

子网络创建完成后，通过如下代码查看网络的参数量：
# build2 个子网络，并打印网络参数信息
conv_net.build(input_shape=[4, 32, 32, 3])
fc_net.build(input_shape=[4, 512])
conv_net.summary()
fc_net.summary()

卷积网络总参数量约为 940 万个，全连接网络总参数量约为 17.7 万个，网络总参数量约为950 万个，相比于原始版本的 VGG13 参数量减少了很多。
由于我们将网络实现为 2 个子网络，在进行梯度更新时，需要合并 2 个子网络的待优化参数列表。代码如下：
# 列表合并，合并 2 个子网络的参数
variables = conv_net.trainable_variables + fc_net.trainable_variables
# 对所有参数求梯度
grads = tape.gradient(loss, variables)
# 自动更新
optimizer.apply_gradients(zip(grads, variables))

运行 cifar10_train.py 文件即可开始训练模型，在训练完 50 个 Epoch 后，网络的测试准确率达到了 77.5%。
"""