# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/25 10:55
"""
本节我们基于 VAE 模型实战 Fashion MNIST 图片的重建与生成。如图 12.13 所示，输入为 Fashion MNIST 图片向量，
经过 3 个全连接层后得到隐向量𝐳的均值与方差，分别用两个输出节点数为 20 的全连接层表示，FC2 的 20 个输出节点表示 20 个特征分布的均值向量 ，
FC3 的 20 个输出节点表示 20 个特征分布的取log后的方差向量。通过Reparameterization Trick 采样获得长度为 20 的隐向量𝐳，并通过 FC4 和 FC5 重建出样本图片。

VAE 作为生成模型，除了可以重建输入样本，还可以单独使用解码器生成样本。通过从先验分布𝑝(𝐳)中直接采样获得隐向量𝐳，经过解码后可以产生生成的样本。

12.5.1VAE模型
我们将 Encoder 和 Decoder 子网络实现在 VAE 大类中，在初始化函数中，分别创建Encoder 和 Decoder 需要的网络层。
代码如下：
class VAE(keras.Model):
    # 变分自编码器
        def __init__(self):
            super(VAE, self).__init__()

            # Encoder 网络
            self.fc1 = layers.Dense(128)
            self.fc2 = layers.Dense(z_dim) # 均值输出
            self.fc3 = layers.Dense(z_dim) # 方差输出

            # Decoder 网络
            self.fc4 = layers.Dense(128)
            self.fc5 = layers.Dense(784)
Encoder 的输入先通过共享层 FC1，然后分别通过 FC2 与 FC3 网络，获得隐向量分布的均值向量与方差的log向量值。
代码如下：
        def encoder(self, x):
            # 获得编码器的均值和方差
            h = tf.nn.relu(self.fc1(x))
            # 均值向量
            mu = self.fc2(h)
            # 方差的 log 向量
            log_var = self.fc3(h)

            return mu, log_var
Decoder 接受采样后的隐向量𝐳，并解码为图片输出。
代码如下：
        def decoder(self, z):

            # 根据隐藏变量 z 生成图片数据
            out = tf.nn.relu(self.fc4(z))
            out = self.fc5(out)
            # 返回图片数据，784 向量
            return out
在 VAE 的前向计算过程中，首先通过编码器获得输入的隐向量𝐳的分布，然后利用Reparameterization Trick 实现的 reparameterize 函数采样获得隐向量𝐳，
最后通过解码器即可恢复重建的图片向量。
实现如下：
        def call(self, inputs, training=None):
            # 前向计算
            # 编码器[b, 784] => [b, z_dim], [b, z_dim]
            mu, log_var = self.encoder(inputs)
            # 采样 reparameterization trick
            z = self.reparameterize(mu, log_var)
            # 通过解码器生成
            x_hat = self.decoder(z)
            # 返回生成样本，及其均值与方差
            return x_hat, mu, log_var

12.5.2Reparameterization技巧
Reparameterize 函数接受均值与方差参数，并从正态分布𝒩(0,𝐼)中采样获得𝜀，通过 z = 𝜇 + 𝜎 ⊙ 𝜀方式返回采样隐向量。
代码如下：
        def reparameterize(self, mu, log_var):
            # reparameterize 技巧，从正态分布采样 epsion
            eps = tf.random.normal(log_var.shape)
            # 计算标准差
            std = tf.exp(log_var)**0.5
            # reparameterize 技巧
            z = mu + std * eps
            return z

12.5.3网络训练
网络固定训练 100 个 Epoch，每次从 VAE 模型中前向计算获得重建样本，通过交叉熵损失函数计算重建误差项𝔼𝒛~𝑞[𝑙𝑜𝑔 𝑝𝜃(𝒙|𝒛)]，
根据公式(12-2)计算𝔻𝐾𝐿 (𝑞 (𝒛|𝒙)||𝑝(𝒛))误差项，并自动求导和更新整个网络模型。代码如下
# 创建网络对象
model = VAE()
model.build(input_shape=(4, 784))
# 优化器
optimizer = optimizers.Adam(lr)
for epoch in range(100): # 训练 100 个 Epoch
    for step, x in enumerate(train_db): # 遍历训练集
        # 打平，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # 构建梯度记录器
        with tf.GradientTape() as tape:
            # 前向计算
            x_rec_logits, mu, log_var = model(x)
            # 重建损失值计算
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            # 计算 KL 散度 N(mu, var) VS N(0, 1)
            # 公式参考：https://stats.stackexchange.com/questions/7440/kldivergence-between-two-univariate-gaussians
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


12.5.4图片生成
图片生成只利用到解码器网络，首先从先验分布𝒩(0,𝐼)中采样获得隐向量，再通过解码器获得图片向量，最后 Reshape 为图片矩阵。
例如：
# 测试生成效果，从正态分布随机采样 z
z = tf.random.normal((batchsz, z_dim))
logits = model.decoder(z) # 仅通过解码器生成图片
x_hat = tf.sigmoid(logits) # 转换为像素范围
x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() *255.
x_hat = x_hat.astype(np.uint8)
save_images(x_hat, 'vae_images/epoch_%d_sampled.png'%epoch) # 保存生成图片

# 重建图片，从测试集采样图片
x = next(iter(test_db))
logits, _, _ = model(tf.reshape(x, [-1, 784])) # 打平并送入自编码器
x_hat = tf.sigmoid(logits) # 将输出转换为像素值
# 恢复为 28x28,[b, 784] => [b, 28, 28]
x_hat = tf.reshape(x_hat, [-1, 28, 28])
# 输入的前 50 张+重建的前 50 张图片合并，[b, 28, 28] => [2b, 28, 28]
x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
x_concat = x_concat.numpy() * 255. # 恢复为 0~255 范围
x_concat = x_concat.astype(np.uint8)
save_images(x_concat, 'vae_images/epoch_%d_rec.png'%epoch) # 保存重建图片

图片重建的效果如图 12.15、图 12.16、图 12.17 所示，分别显示了在第 1、10、100个 Epoch 时，输入测试集的图片，获得的重建效果，
每张图片的左 5 列为真实图片，右 5列为对应的重建效果。图片的生成效果图 12.18、图 12.19、图 12.20 所示，分别显示了在第 1、10、100 个 Epoch 时，
图片的生成效果。
"""