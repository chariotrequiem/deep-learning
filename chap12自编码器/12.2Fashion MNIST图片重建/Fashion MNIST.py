# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/24 22:30
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt


h_dim = 20
batchsz = 128
lr = 1e-3


def save_images(imgs, name):
    # 创建 280x280 大小图片阵列
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):  # 10 行图片阵列
        for j in range(0, 280, 28):  # 10 列图片阵列
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))  # 写入对应位置
            index += 1  # 保存图片阵列
    new_im.save(name)


# 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 数据归一化
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# 只需要通过图片数据即可构建数据集对象，不需要标签
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
# 构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)


# 自编码器
class AE(keras.Model):
    # 自编码器模型类，包含了 Encoder 和 Decoder2 个子网络
    def __init__(self):
        super(AE, self).__init__()
        # 创建 Encoders 网络
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        # 创建 Decoders 网络
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # 前向传播函数
        # 编码获得隐藏向量 h,[b, 784] => [b, 20]
        h = self.encoder(inputs)
        # 解码获得重建图片，[b, 20] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


model = AE()
model.build(input_shape=(None, 784))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)

for epoch in range(100):

    for step, x in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

    # 重建图片，从测试集采样一批图片
    x = next(iter(test_db))
    logits = model(tf.reshape(x, [-1, 784]))  # 打平并送入自编码器
    x_hat = tf.sigmoid(logits)  # 将输出转换为像素值，使用 sigmoid 函数
    # 恢复为 28x28,[b, 784] => [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])
    # 输入的前 50 张+重建的前 50 张图片合并，[b, 28, 28] => [2b, 28, 28]
    x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
    x_concat = x_concat.numpy() * 255.  # 恢复为 0~255 范围
    x_concat = x_concat.astype(np.uint8)  # 转换为整型
    save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)  # 保存图片


