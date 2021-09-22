# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 16:09
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics

# 创建5层的全连接网络，用于MNIST手写数字图片识别
# 首先创建5层的全连接神经网络
network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(4, 28*28))
print(network.summary())

# 模型装配
# 采用Adam优化器，学习率为0.01；采用交叉熵损失函数，包含Softmax
network.compile(optimizer=optimizers.Adam(lr=0.01),
                # from_logits: True就是需要经过softmax进行概率化，默认False及y_pred是经过softmax处理的
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


# 数据预处理
def preprocess(x, y):
    # [b, 28, 28], [b]
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)  # one_hot编码

    return x, y


# 加载数据
# 加载mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建Dataset对象
train_db = train_db.shuffle(60000)  # 将对象随机打散
train_db = train_db.batch(128)  # 批训练，每批训练128张图片
train_db = train_db.map(preprocess)  # 进行预处理

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 创建Dataset对象
test_db = test_db.shuffle(60000).batch(128).map(preprocess)

# 指定训练集为 train_db，验证集为 val_db,训练 5 个 epochs，每 2 个 epoch 验证一次
# 返回训练轨迹信息保存在 history 对象中
history = network.fit(train_db, epochs=5, validation_data=test_db, validation_freq=2)

# 加载一个batch的测试数据
x, y = next(iter(test_db))
print('predict x: ', x.shape)  # 打印当前batch的形状
pred = network.predict(x)  # 模型预测，预测结果保存在out中
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)

correct = tf.equal(y, pred)
total_correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
print('acc = {}%'.format(total_correct/128*100))
# print('acc = {}'.format(sum / 128))

# 模型测试，测试在 db_test 上的性能表现
network.evaluate(test_db)

# 保存模型结构与模型参数到文件
tf.saved_model.save(network, 'model-savedmodel')
print('saving savedmodel.')
del network  # 删除网络对象

print('load savedmodel from file.')  # 从文件恢复网络结构与网络参数
network = tf.saved_model.load('model-savedmodel')  # 准确率计量器
acc_meter = metrics.CategoricalAccuracy()
for x, y in test_db:  # 遍历测试集
    pred = network(x)  # 前向计算
    acc_meter.update_state(y_true=y, y_pred=pred)  # 更新准确率统计
# 打印准确率
print("Test Accuracy:%f" % acc_meter.result())


