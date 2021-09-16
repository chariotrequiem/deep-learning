# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 22:10
import matplotlib
from matplotlib import pyplot as plt

# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)


# 数据预处理
def preprocess(x, y):
    # [b, 28, 28], [b]
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)  # one_hot编码

    return x, y


# 加载mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
# 将数据转换为Dataset对象，经过数据集标准处理步骤：随机打散，预处理，按批装载等
batchsz = 128  # 每批样本数量,即一次并行计算 128 个样本的数据
train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建Dataset对象
train_db = train_db.shuffle(10000)  # 将对象随机打散
train_db = train_db.batch(batchsz)  # 批训练，每批训练128张图片
train_db = train_db.map(preprocess)  # 进行预处理
train_db = train_db.repeat(20)  # 数据集迭代20次才终止

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 创建Dataset对象
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)


# print(x[0], y[0])


# %%
def main():
    # 设置学习率
    lr = 1e-2
    accs, losses = [], []

    # 设置第一层的参数w1, b1 784 => 256
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 设置第二层的参数w2, b2 256 => 128
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 设置第三层的参数w3, b3 128 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    # 循环训练
    for step, (x, y) in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:

            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # 计算损失
            # [b, 10] - [b, 10]
            loss = tf.square(y - out)
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        # print
        if step % 250 == 0:
            print(step, 'loss:', float(loss))
            losses.append(float(loss))

        if step % 250 == 0:
            # evaluate/test
            total, total_correct = 0., 0

            for x, y in test_db:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)  # 选取概率最大的类别
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)  # one_hot编码逆过程
                # bool type
                correct = tf.equal(pred, y)  # 比较预测值与真实值
                # bool tensor => int tensor => numpy
                # 统计预测正确的样本个数
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)

            accs.append(total_correct / total)

    plt.figure()
    x = [i * 250 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')

    plt.show()


if __name__ == '__main__':
    main()