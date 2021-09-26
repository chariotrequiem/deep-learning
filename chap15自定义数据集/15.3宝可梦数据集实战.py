# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/26 16:39
"""
在介绍完自定义数据集的加载流程后，我们来实战宝可梦数据集的加载以及训练。

15.3.1创建Dataset对象
首先通过 load_pokemon 函数返回 images、labels 和编码表信息，代码如下：
# 加载 pokemon 数据集，指定加载训练集
# 返回训练集的样本路径列表，标签数字列表和编码表字典
images, labels, table = load_pokemon('pokemon', 'train')
print('images:', len(images), images)
print('labels:', len(labels), labels)
print('table:', table)
构建 Dataset 对象，并完成数据集的随机打散、预处理和批量化操作，代码如下：
# images: string path
# labels: number
db = tf.data.Dataset.from_tensor_slices((images, labels))
db = db.shuffle(1000).map(preprocess).batch(32)
我们在使用 tf.data.Dataset.from_tensor_slices 构建数据集时传入的参数是 images 和 labels 组成的 tuple，因此在对 db 对象迭代时，
返回的是(𝑿𝑖, 𝒀𝑖)的 tuple 对象，其中𝑿𝑖是第𝑖 个 Batch 的图片张量数据，𝒀𝑖是第𝑖个 Batch 的图片标签数据。我们可以通过 TensorBoard
可视化来查看每次遍历的图片样本，代码如下：
# 创建 TensorBoard summary 对象
writter = tf.summary.create_file_writer('logs')
for step, (x,y) in enumerate(db):
    # x: [32, 224, 224, 3]
    # y: [32]
    with writter.as_default():
        x = denormalize(x) # 反向 normalize，方便可视化
        # 写入图片数据
        tf.summary.image('img',x,step=step,max_outputs=9)
        time.sleep(5) # 延迟 5s，再此循环

15.3.2数据预处理
上面我们在构建数据集时通过调用.map(preprocess)函数来完成数据的预处理工作。由于目前我们的 images 列表只是保存了所有图片的路径信息，
而不是图片的内容张量，因此需要在预处理函数中完成图片的读取以及张量转换等工作。

对于预处理函数(x,y) = preprocess(x,y)，它的传入参数需要和创建 Dataset 时给的参数的格式保存一致，返回参数也需要和传入参数的格式保存一致。
特别地，我们在构建数据集时传入(𝒙, 𝒚)的 tuple 对象，其中𝒙为所有图片的路径列表，𝒚为所有图片的标签数字列表。
考虑到 map 函数的位置为 db = db.shuffle(1000).map(preprocess).batch(32)，那么preprocess 的传入参数为(𝑥𝑖, 𝑦𝑖)，
其中𝑥𝑖和𝑦𝑖分别为第𝑖个图片的路径字符串和标签数字。如 果 map 函数的位置为 db = db.shuffle(1000).batch(32) .map(preprocess)，
那么 preprocess 的传入参数为(𝒙𝑖, 𝒚𝑖)，其中𝒙𝑖和𝒚𝑖分别为第𝑖个 Batch 的路径和标签列表。
代码如下：
def preprocess(x,y): # 预处理函数
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x) # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # 图片解码，忽略透明通道
    x = tf.image.resize(x, [244, 244]) # 图片缩放为略大于 224 的 244
    # 数据增强，这里可以自由组合增强手段
    # x = tf.image.random_flip_up_down(x)
    x = tf.image.random_flip_left_right(x) # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3]) # 随机裁剪为 224
    # 转换成张量，并压缩到 0~1 区间
    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255. # 0~1 => D(0,1)
    x = normalize(x) # 标准化
    y = tf.convert_to_tensor(y) # 转换成张量

    return x, y

考虑到我们的数据集规模非常小，为了防止过拟合，我们做了少量的数据增强变换，以获得更多样式的图片数据。最后我们将 0~255 范围的像素值缩放到 0~1 范围，
并通过标准化函数 normalize 实现数据的标准化运算，将像素映射为 0 周围分布，有利于网络的优化。最后将数据转换为张量数据返回。
此时对 db 对象迭代时返回的数据将是批量形式的图片张量数据和标签张量。

标准化后的数据适合网络的训练及预测，但是在进行可视化时，需要将数据映射回0~1 的范围。实现标准化和标准化的逆过程如下：
# 这里的 mean 和 std 根据真实的数据计算获得，比如 ImageNet
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])

    def normalize(x, mean=img_mean, std=img_std):
    # 标准化函数
    # x: [224, 224, 3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean)/std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    # 标准化的逆过程函数
    x = x * std + mean
    return x

使用上述方法，分布创建训练集、验证集和测试集的 Dataset 对象。一般来说，验证集和测试集并不直接参与网络参数的优化，并不需要随机打散样本次序。
代码如下：
batchsz = 128
# 创建训练集 Dataset 对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集 Dataset 对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集 Dataset 对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)


15.3.3创建模型
前面已经介绍并实现了 VGG13 和 ResNet18 等主流网络模型，这里我们就不再赘述模型的具体实现细节。
在 keras.applications 模块中实现了常用的网络模型，如 VGG 系列、 ResNet 系列、DenseNet 系列、MobileNet 系列等等，
只需要一行代码即可创建这些模型网络。例如：
# 加载 DenseNet 网络模型，并去掉最后一层全连接层，最后一个池化层设置为 max pooling
net = keras.applications.DenseNet121(include_top=False, pooling='max')
# 设置为 True，即 DenseNet 部分的参数也参与优化更新
net.trainable = True

newnet = keras.Sequential([
    net, # 去掉最后一层的 DenseNet121
    layers.Dense(1024, activation='relu'), # 追加全连接层
    layers.BatchNormalization(), # 追加 BN 层
    layers.Dropout(rate=0.5), # 追加 Dropout 层，防止过拟合
    layers.Dense(5) # 根据宝可梦数据的类别数，设置最后一层输出节点数为 5
])
newnet.build(input_shape=(4,224,224,3))
newnet.summary()

上面使用 DenseNet121 模型来创建网络，由于 DenseNet121 的最后一层输出节点设计为1000，我们将 DenseNet121 去掉最后一层，
并根据自定义数据集的类别数，添加一个输出节点数为 5 的全连接层，通过 Sequential 容器重新包裹成新的网络模型。
其中include_top=False 表明去掉最后的全连接层，pooling='max'表示 DenseNet121 最后一个Pooling 层设计为 Max Polling。
网络模型结构图 15.4 所示。

15.3.4网络测试与训练
我们直接使用 Keras 提供的 Compile&Fit 方式装配并训练网络，优化器采用最常用的Adam 优化器，误差函数采用交叉熵损失函数，
并设置 from_logits=True，在训练过程中关注的测量指标为准确率。网络模型装配代码如下：
# 装配模型
newnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
通过 fit 函数在训练集上面训练模型，每迭代一个 Epoch 测试一次验证集，最大训练Epoch 数为 100，为了防止过拟合，我们采用了 Early Stopping 技术，
在 fit 函数的 callbacks参数中传入 Early Stopping 类实例。
代码如下：
# 训练模型，支持 early stopping
history = newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100, callbacks=[early_stopping])
其中 early_stopping 为标准的 EarlyStopping 类，它监听的指标是验证集准确率，如果连续三次验证集的测量结果没有提升 0.001，
则触发 EarlyStopping 条件，训练结束。代码如下：

# 创建 Early Stopping 类，连续 3 次不上升则终止训练
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=3
)
我们将训练过程中的训练准确率、验证准确率以及最后测试集上面获得的准确率绘制为曲线，如图 15.5 所示。可以看到，训练准确率迅速提升并维持在较高状态，
但是验证准确率比较差，同时并没有获得较大提升，Early Stopping 条件粗发，训练很快终止，网络出现了非常严重的过拟合现象。

那么为什么会出现过拟合现象呢？考虑我们使用的 DensetNet121 模型的层数达到了121 层，参数量达到了 7M 个，是比较大型的网络模型，
而我们大数据集仅有约 1000 个样本。根据经验，这远远不足以训练好如此大规模的网络模型，极其容易出现过拟合现象。
为了减轻过拟合，可以采用层数更浅、参数量更少的网络模型，或者添加正则化项，甚至增加数据集的规模等。
除了这些方式以外，另外一种行之有效的方式就是迁移学习技术。

"""