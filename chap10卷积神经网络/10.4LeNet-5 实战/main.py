# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 21:42
import Data
import tensorflow as tf
# 导入误差计算，优化器模块
from tensorflow.keras import Sequential, datasets, layers, losses

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# 加载mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# 对数据进行预处理，分割成训练集与测试集
train_db = Data.db(x, y, 60000, 128)
test_db = Data.db(x_test, y_test, 10000, 128)
print(train_db, test_db)

network = Sequential([  # 网络容器
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层，6个3*3卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Flatten(),  # 打平层，方便全连接层处理

    layers.Dense(120, activation='relu'),  # 全连接层，120 个节点
    layers.Dense(84, activation='relu'),  # 全连接层，84 节点
    layers.Dense(10)  # 全连接层，10 个节点
])

# build一次网络模型，给输入X的形状，其中4为随意给的batchsz
network.build(input_shape=(128, 28, 28, 1))
# 统计网络信息
network.summary()

# 创建损失函数的类，在实际计算时直接调用类实例即可
criteon = losses.CategoricalCrossentropy(from_logits=True)


# 训练部分实现如下
optimizer = tf.keras.optimizers.RMSprop(0.001)  # 创建优化器，指定学习率

for epoch in range(20):  # 20个Epoch
    print('epoch{}'.format(epoch))
    for step, (x, y) in enumerate(train_db):  # 遍历一次训练集
        # print(x.shape, y.shape)
        # 构建梯度记录环境
        with tf.GradientTape() as tape:
            # 插入通道维度，=>[b,28,28,1]
            x = tf.expand_dims(x, axis=3)
            # 前向计算，获得 10 类别的概率分布，[b, 784] => [b, 10]
            out = network(x)
            # 真实标签 one-hot 编码，[b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)  # 计算交叉熵损失函数，标量
            loss = criteon(y_onehot, out)
            if step % 200 == 0:
                print('loss=' + str(loss))

        # 自动计算梯度
        grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新参数
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

# 记录预测正确的数量，总样本数量
correct, total = 0, 0
for x, y in test_db:  # 遍历所有训练集样本
    # 插入通道维度，=>[b,28,28,1]
    x = tf.expand_dims(x, axis=3)  # 前向计算，获得 10 类别的预测分布，[b, 784] => [b, 10]
    out = network(x)
    # 真实的流程时先经过 softmax，再 argmax
    # 但是由于 softmax 不改变元素的大小相对关系，故省去
    pred = tf.argmax(out, axis=-1)
    y = tf.cast(y, tf.int64)
    # 统计预测正确数量
    correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
    # 统计预测样本总数
    total += x.shape[0]  # 计算准确率
print('test acc:', correct/total)
