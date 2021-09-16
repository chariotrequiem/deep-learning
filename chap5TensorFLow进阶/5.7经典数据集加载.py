# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 21:35
"""
到这里为止，我们已经学习完张量的常用操作方法，已具备实现大部分深度网络的技术储备。
最后我们将以一个完整的张量方式实现的分类网络模型实战收尾本章。在进入实战之前，我们先正式介绍对于常用的经典数据集，
如何利用 TensorFlow 提供的工具便捷地加载数据集。对于自定义的数据集的加载，我们会在后续章节介绍。

在 TensorFlow 中，keras.datasets 模块提供了常用经典数据集的自动下载、管理、加载与转换功能，
并且提供了 tf.data.Dataset 数据集对象，方便实现多线程(Multi-threading)、预处理(Preprocessing)、
随机打散(Shuffle)和批训练(Training on Batch)等常用数据集的功能。对于常用的经典数据集，
例如：
❑ Boston Housing，波士顿房价趋势数据集，用于回归模型训练与测试。
❑ CIFAR10/100，真实图片数据集，用于图片分类任务。
❑ MNIST/Fashion_MNIST，手写数字图片数据集，用于图片分类任务。
❑ IMDB，情感分类任务数据集，用于文本分类任务。

这些数据集在机器学习或深度学习的研究和学习中使用的非常频繁。对于新提出的算法，一般优先在经典的数据集上面测试，
再尝试迁移到更大规模、更复杂的数据集上。

通过 datasets.xxx.load_data()函数即可实现经典数据集的自动加载，其中 xxx 代表具体的数据集名称，
如“CIFAR10”、“MNIST”。TensorFlow 会默认将数据缓存在用户目录下的.keras/datasets 文件夹，如图 5.6 所示，
用户不需要关心数据集是如何保存的。如果当前数据集不在缓存中，则会自动从网络下载、解压和加载数据集；
如果已经在缓存中，则自动完成加载。
例如，自动加载 MNIST 数据集：
In [66]:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets # 导入经典数据集加载模块
# 加载 MNIST 数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
Out [66]: # 返回数组的形状
x: (60000, 28, 28) y: (60000,) x test: (10000, 28, 28) y test: [7 2 1 ... 4 5 6]

通过 load_data()函数会返回相应格式的数据，对于图片数据集 MNIST、CIFAR10 等，会返回 2 个 tuple，
第一个 tuple 保存了用于训练的数据 x 和 y 训练集对象；第 2 个 tuple 则保存了用于测试的数据 x_test 和 y_test 测试集对象，
所有的数据都用 Numpy 数组容器保存。

数据加载进入内存后，需要转换成 Dataset 对象，才能利用 TensorFlow 提供的各种便捷功能。
通过 Dataset.from_tensor_slices 可以将训练部分的数据图片 x 和标签 y 都转换成Dataset 对象：
train_db = tf.data.Dataset.from_tensor_slices((x, y)) # 构建 Dataset 对象
将数据转换成 Dataset 对象后，一般需要再添加一系列的数据集标准处理步骤，如随机打散、预处理、按批装载等。


5.7.1随机打散
通过 Dataset.shuffle(buffer_size)工具可以设置 Dataset 对象随机打散数据之间的顺序，
防止每次训练时数据按固定顺序产生，从而使得模型尝试“记忆”住标签信息，代码实现
如下：
train_db = train_db.shuffle(10000) # 随机打散样本，不会打乱样本与标签映射关系
其中，buffer_size 参数指定缓冲池的大小，一般设置为一个较大的常数即可。
调用 Dataset提供的这些工具函数会返回新的 Dataset 对象，可以通过
                   db = db. step1(). step2(). step3.()
方式按序完成所有的数据处理步骤，实现起来非常方便。


5.7.2批训练
为了利用显卡的并行计算能力，一般在网络的计算过程中会同时计算多个样本，我们把这种训练方式叫做批训练，
其中一个批中样本的数量叫做 Batch Size。为了一次能够从Dataset 中产生 Batch Size 数量的样本，
需要设置 Dataset 为批训练方式，实现如下：
train_db = train_db.batch(128) # 设置批训练，batch size 为 128
其中 128 为 Batch Size 参数，即一次并行计算 128 个样本的数据。Batch Size 一般根据用户的 GPU 显存资源来设置，
当显存不足时，可以适量减少 Batch Size 来减少算法的显存使用量。


5.7.3预处理
从 keras.datasets 中加载的数据集的格式大部分情况都不能直接满足模型的输入要求，因此需要根据用户的逻辑自行实现预处理步骤。
Dataset 对象通过提供 map(func)工具函数，可以非常方便地调用用户自定义的预处理逻辑，它实现在 func 函数里。
例如，下方代码调用名为 preprocess 的函数完成每个样本的预处理：
# 预处理函数实现在 preprocess 函数中，传入函数名即可
train_db = train_db.map(preprocess)

考虑 MNIST 手写数字图片，从 keras.datasets 中经.batch()后加载的图片 x shape 为 [𝑏, 28,28]，像素使用 0~255 的整型表示；
标签 shape 为[𝑏]，即采样数字编码方式。实际的神经网络输入，一般需要将图片数据标准化到[0,1]或[−1,1]等 0 附近区间，
同时根据网络的设置，需要将 shape 为[28,28]的输入视图调整为合法的格式；对于标签信息，可以选择在预处理时进行 One-hot 编码，
也可以在计算误差时进行 One-hot 编码。
根据下一节的实战设定，我们将 MNIST 图片数据映射到𝑥 ∈ [0,1]区间，视图调整为[𝑏, 28 ∗ 28]；对于标签数据，
我们选择在预处理函数里面进行 One-hot 编码。preprocess 函数实现如下：

def preprocess(x, y): # 自定义的预处理函数
    # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28]) # 打平
    y = tf.cast(y, dtype=tf.int32) # 转成整型张量
    y = tf.one_hot(y, depth=10) # one-hot 编码
    # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能
    return x,y


5.7.4循环训练
对于 Dataset 对象，在使用时可以通过
 for step, (x,y) in enumerate(train_db): # 迭代数据集对象，带 step 参数
或
 for x,y in train_db: # 迭代数据集对象
方式进行迭代，每次返回的 x 和 y 对象即为批量样本和标签。当对 train_db 的所有样本完成一次迭代后，
for 循环终止退出。这样完成一个 Batch 的数据训练，叫做一个 Step；通过多个 step 来完成整个训练集的一次迭代，
叫做一个 Epoch。在实际训练时，通常需要对数据集迭代多个 Epoch 才能取得较好地训练效果。
例如，固定训练 20 个 Epoch，实现如下：
for epoch in range(20): # 训练 Epoch 数
    for step, (x,y) in enumerate(train_db): # 迭代 Step 数
        # training...

此外，也可以通过设置 Dataset 对象，使得数据集对象内部遍历多次才会退出，实现如下:
train_db = train_db.repeat(20) # 数据集迭代 20 遍才终止
上述代码使得 for x,y in train_db 循环迭代 20 个 epoch 才会退出。不管使用上述哪种方式，都能取得一样的效果。
由于上一章已经完成了前向计算实战，此处我们略过。
"""