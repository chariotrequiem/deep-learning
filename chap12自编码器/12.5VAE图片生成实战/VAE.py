# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/25 15:45
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

h_dim = 20
batchsz = 512
lr = 1e-3
z_dim = 10


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


class VAE(keras.Model):
    # 变分自编码器
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder 网络
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim) # get mean prediction
        self.fc3 = layers.Dense(z_dim)

        # Decoder 网络
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        # 获得编码器的均值和方差
        h = tf.nn.relu(self.fc1(x))
        # get mean均值向量
        mu = self.fc2(h)
        # get variance 方差的log向量
        log_var = self.fc3(h)

        return mu, log_var

    def decoder(self, z):
        # 根据隐藏变量z生成图片数据
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        # 返回图片数据， 784向量
        return out

    def reparameterize(self, mu, log_var):
        # reparameterize 技巧，从正态分布采样 epsion
        eps = tf.random.normal(log_var.shape)
        # 计算标准差
        std = tf.exp(log_var*0.5)
        # reparameterize 技巧
        z = mu + std * eps
        return z

    def call(self, inputs, training=None):
        # 前向计算
        # 编码器 [b, 784] => [b, z_dim], [b, z_dim]
        mu, log_var = self.encoder(inputs)
        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        # 通过解码器生成
        x_hat = self.decoder(z)
        # 返回生成样本，及其均值与方差
        return x_hat, mu, log_var


# 加载数据， 数据预处理
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 创建网络对象
model = VAE()
model.build(input_shape=(4, 784))
# 优化器
optimizer = tf.optimizers.Adam(lr)

for epoch in range(500):

    for step, x in enumerate(train_db):
        # 打平，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # 构建梯度记录器
        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)  # 前向计算
            # 重建损失值
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

            # 计算 kl 散度 (mu, var) ~ N (0, 1)
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu**2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            # 合并误差项
            loss = rec_loss + 1. * kl_div
        # 自动求导
        grads = tape.gradient(loss, model.trainable_variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            # 打印训练误差
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))

    # evaluation
    # 测试生成效果，从正态分布随机采样 z
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)  # 仅通过解码器生成图片
    x_hat = tf.sigmoid(logits)  # 转换为像素范围
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/epoch_%d_sampled.png' % epoch)  # 保存生成图片
    # 重建图片，从测试集采样图片
    x = next(iter(test_db))
    logits, _, _ = model(tf.reshape(x, [-1, 784]))  # 打平并送入自编码器
    x_hat = tf.sigmoid(logits)  # 将输出转换为像素值
    # 恢复为 28x28,[b, 784] => [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])
    # 输入的前 50 张+重建的前 50 张图片合并，[b, 28, 28] => [2b, 28, 28]
    x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
    x_concat = x_concat.numpy() * 255.  # 恢复为 0~255 范围
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat, 'vae_images/epoch_%d_rec.png' % epoch)  # 保存重建图片
