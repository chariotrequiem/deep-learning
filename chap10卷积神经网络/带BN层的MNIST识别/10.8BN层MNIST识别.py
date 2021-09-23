# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 10:57
import tensorflow as tf
from tensorflow.keras import Sequential, datasets, layers, losses
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# 数据预处理
def preprocess(x, y):
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.  # 缩放到0-1
    # x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)  # y转换为整型
    # y = tf.one_hot(y, depth=10)  # one_hot编码
    return x, y


# 加载mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# 对数据进行预处理，创建训练集和测试集
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(60000).batch(100).map(preprocess)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000).batch(100).map(preprocess)
print(train_db, test_db)


network = Sequential([  # 网络容器
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层，6个3*3卷积核
    # 插入Bn层
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
    # 插入BN层
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Flatten(),  # 打平层，方便全连接层处理

    layers.Dense(120, activation='relu'),  # 全连接层，120 个节点
    # 此处也可以插入BN层
    layers.Dense(84, activation='relu'),  # 全连接层，84 节点
    # 此处也可以插入BN层
    layers.Dense(10)  # 全连接层，10 个节点
])

# build一次网络模型，给输入X的形状，其中4为随意给的batchsz
network.build(input_shape=(100, 28, 28, 1))
# 统计网络信息
network.summary()

# 创建损失函数的类，在实际计算时，直接调用类实例即可
# from_logits:True就是需要经过softmax进行概率化，默认False及y_pred是经过softmax处理的
criteon = losses.CategoricalCrossentropy(from_logits=True)

# 训练部分实现
optimizer = tf.keras.optimizers.RMSprop(0.001)  # 创建优化器，使用RMSprop优化办法，指定学习率
accs = []
for epoch in range(20):  # 训练20个epoch
    total, total_correct = 0, 0
    for step, (x, y) in enumerate(train_db):  # 遍历训练集
        with tf.GradientTape() as tape:
            # 插入通道维度
            x = tf.expand_dims(x, axis=3)
            # 前向计算，设置计算模式
            out = network(x, training=True)  # 设置参数training=True代表BN层是训练
            # 真实标签 one-hot 编码，[b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)  # 计算交叉熵损失函数，标量
            loss = criteon(y_onehot, out)

        if step % 20 == 0:  # 间歇性打印训练误差
            print('epoch: {}  step: {}  loss: {}'.format(epoch+1, step, loss))

        # 自动计算梯度
        grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新参数
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

    pred = tf.argmax(out, axis=1)  # 选取概率最大的类别
    # 将y转化为int64的数值
    y = tf.cast(y, tf.int64)
    # correct = tf.equal(pred, y)  # 比较预测值与真实值
    # bool tensor => int tensor => numpy
    # 统计预测正确的样本个数
    total_correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
    total += x.shape[0]
    acc = total_correct / total * 100
    accs.append(acc)
    print('第{}个epoch的准确率为{}%'.format(epoch + 1, acc))


# 在测试集中测试准确率
# 记录预测正确的数量，总样本数量
correct, total = 0, 0
for step, (x, y) in enumerate(test_db):  # 遍历所有训练集样本
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

print('test acc: {}%'.format(correct/total*100))

# 训练集准确率可视化
x = [i for i in range(0, 20)]
fig = plt.figure(figsize=(8, 6))
plt.title('accuracy in train_db')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([x for x in range(21) if x % 1 == 0])  # x标记step设置为1
plt.plot(x, accs, color='red')
plt.show()

