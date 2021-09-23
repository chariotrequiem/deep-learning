# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 16:11
import tensorflow as tf
from tensorflow.keras import Sequential, datasets, layers, losses
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# 预处理函数
def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32)/255. - 1
    y = tf.cast(y, dtype=tf.int32)  # 将y从uint8型转换为int32型
    return x, y


# 在线下载，加载CIFAR10数据集
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
# 删除 y 的一个维度，[b,1] => [b]
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
# 打印训练集和测试集的形状
print(x.shape, y.shape, x_test.shape, y_test.shape)  # (50000, 32, 32, 3) (50000,) (10000, 32, 32, 3) (10000,)
# 构建训练集对象，随机打乱、预处理、批量化
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)
# 构建测试集对象，预处理，批量化
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)
# 从训练集中采样一个Batch，并观察
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


conv_layers = [  # 先创建包含多网络层的列表
    # Conv-Conv-Pooling单元1
    # 64个3*3卷积核，输入输出同大小
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # strides默认值为1
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
# [b, 32, 32, 3] => [b, 1, 1, 512]
conv_net = Sequential(conv_layers)

# 创建3层全连接层子网络
# 包含3个全连接层，每层添加ReLU非线性激活函数，最后一层除外
fc_net = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])

# 创建子网络
# build2个子网络，并打印网络参数信息
conv_net.build(input_shape=[None, 32, 32, 3])
fc_net.build(input_shape=[None, 512])
conv_net.summary()
fc_net.summary()

# 列表合并，合并2个子网络的参数
variables = conv_net.trainable_variables + fc_net.trainable_variables


# 训练部分实现
optimizer = tf.keras.optimizers.RMSprop(0.001)  # 创建优化器，使用RMSprop优化办法，指定学习率

for epoch in range(30):  # 训练30个epoch
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            out = conv_net(x)
            # flatten, => [b, 512]
            out = tf.reshape(out, [-1, 512])
            # [b, 512] => [b, 10]
            logits = fc_net(out)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # compute loss
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if step % 10 == 0:
            print("epoch: {} step: {} loss: {}".format(epoch+1, step, float(loss)))

total_num = 0
total_correct = 0
for x, y in test_db:
    out = conv_net(x)
    out = tf.reshape(out, [-1, 512])
    logits = fc_net(out)
    prob = tf.nn.softmax(logits, axis=1)
    pred = tf.argmax(prob, axis=1)
    pred = tf.cast(pred, dtype=tf.int32)

    correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
    correct = tf.reduce_sum(correct)

    total_num += x.shape[0]
    total_correct += int(correct)

acc = total_correct / total_num
print(acc)
# print("epoch: {} acc: {}".format(epoch+1, acc))


