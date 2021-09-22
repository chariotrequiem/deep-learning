# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 10:06
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics


network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(4, 28*28))
network.summary()


network.compile(optimizer=optimizers.Adam(lr=0.01),
                # from_logits: True就是需要经过softmax进行概率化，默认False及y_pred是经过softmax处理的
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


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

x, y = next(iter(test_db))
acc_meter = metrics.Accuracy() # 创建准确率测量器
# [b, 784] => [b, 10]，网络输出值
out = network(x)
# [b, 10] => [b]，经过 argmax 后计算预测值
pred = tf.argmax(out, axis=1)
pred = tf.cast(pred, dtype=tf.int32)
y = tf.argmax(y, axis=1)
# 根据预测值与真实值写入测量器
acc_meter.update_state(y, pred)

# 读取统计结果
print('Evaluate Acc:', acc_meter.result().numpy())
acc_meter.reset_states()  # 清零测量器