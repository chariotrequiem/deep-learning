# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/24 15:07
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
from resnet import resnet18

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    # 将数据映射到-1~1
    x = 2*tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)  # 类型转换
    return x, y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()  # 加载数据集
y = tf.squeeze(y, axis=1)  # 删除不必要的维度
y_test = tf.squeeze(y_test, axis=1)  # 删除不必要的维度
print(x.shape, y.shape, x_test.shape, y_test.shape)


train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建训练集
# 随机打散，预处理，批量化
train_db = train_db.shuffle(20000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构建测试集
# 随机打散，预处理，批量化
test_db = test_db.map(preprocess).batch(128)
# 采样一个样本
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()  # ResNet18网络
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()  # 统计网络参数

    optimizer = optimizers.Adam(lr=1e-4)  # 构建优化器, 学习率为1e-4

    accs = []
    for epoch in range(3):  # 训练epoch
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 10],前向传播
                logits = model(x)
                # [b] => [b, 10],one-hot编码
                y_onehot = tf.one_hot(y, depth=10)
                # 计算交叉熵
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            # 计算梯度信息
            grads = tape.gradient(loss, model.trainable_variables)
            # 更新网络参数
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print("epoch: {}  step: {}  loss: {}".format(epoch, step, float(loss)))

        total_num = 0
        total_correct = 0
        for x, y in test_db:

            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        accs.append(acc)
        print("epoch: {} acc: {}".format(epoch, acc))

    # 训练集准确率可视化
    x = [i for i in range(0, 30)]
    fig = plt.figure(figsize=(8, 6))
    plt.title('accuracy in test_db')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xticks([x for x in range(30) if x % 1 == 0])  # x标记step设置为1
    plt.plot(x, accs, color='red')
    plt.show()


if __name__ == '__main__':
    main()