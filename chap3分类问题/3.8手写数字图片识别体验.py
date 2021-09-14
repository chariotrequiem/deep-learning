# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/13 20:04
"""
本节我们将在未介绍 TensorFlow 的情况下，先带大家体验一下神经网络的乐趣。本节的主要目的并不是教会每个细节，
而是让读者对神经网络算法有全面、直观的感受，为接下来介绍 TensorFlow 基础和深度学习理论打下基础。让我们开始体验神奇的图片识别算法吧！
3.8.1网络搭建
对于第一层模型来说，它接受的输入𝒙 ∈ 𝑅^784，输出𝒉1 ∈ 𝑅^256设计为长度为 256 的向
量，我们不需要显式地编写𝒉1 = ReLU(𝑾1𝒙 + 𝒃1)的计算逻辑，在 TensorFlow 中通过一行代码即可实现：
# 创建一层网络，设置输出节点数为 256，激活函数类型为 ReLU
layers.Dense(256, activation='relu')
使用 TensorFlow 的 Sequential 容器可以非常方便地搭建多层的网络。对于 3 层网络，我们可以通过快速完成 3 层网络的搭建。
# 利用 Sequential 容器封装 3 个网络层，前网络层的输出默认作为下一层的输入
model = keras.Sequential([ # 3 个非线性层的嵌套模型
layers.Dense(256, activation='relu'), # 隐藏层 1
layers.Dense(128, activation='relu'), # 隐藏层 2
layers.Dense(10)]) # 输出层，输出节点数为 10
第 1 层的输出节点数设计为 256，第 2 层设计为 128，输出层节点数设计为 10。
直接调用这个模型对象 model(x)就可以返回模型最后一层的输出𝑜。

3.8.2模型训练
搭建完成 3 层神经网络的对象后，给定输入𝒙，调用 model(𝒙)得到模型输出𝑜后，通过MSE 损失函数计算当前的误差ℒ：
with tf.GradientTape() as tape: # 构建梯度记录环境
    # 打平操作，[b, 28, 28] => [b, 784]
    x = tf.reshape(x, (-1, 28*28))
    # Step1. 得到模型输出 output [b, 784] => [b, 10]
    out = model(x)
    # [b] => [b, 10]
    y_onehot = tf.one_hot(y, depth=10)
    # 计算差的平方和，[b, 10]
    loss = tf.square(out-y_onehot)
    # 计算每个样本的平均误差，[b]
    loss = tf.reduce_sum(loss) / x.shape[0]
再利用 TensorFlow 提供的自动求导函数 tape.gradient(loss, model.trainable_variables)求出模型中所有参数的梯度信息𝜕ℒ/𝜕𝜃 ,
𝜃 ∈ {𝑾1, 𝒃1,𝑾2, 𝒃2,𝑾3, 𝒃3}。
    # Step3. 计算参数的梯度 w1, w2, w3, b1, b2, b3
    grads = tape.gradient(loss, model.trainable_variables)
计算获得的梯度结果使用 grads 列表变量保存。再使用 optimizers 对象自动按照梯度更新法则去更新模型的参数𝜃。
         𝜃′ = 𝜃 − 𝜂 ∙ 𝜕ℒ/𝜕𝜃
实现如下。
     # 自动计算梯度
     grads = tape.gradient(loss, model.trainable_variables)
     # w' = w - lr * grad，更新网络参数
     optimizer.apply_gradients(zip(grads, model.trainable_variables))
循环迭代多次后，就可以利用学好的模型𝑓𝜃去预测未知的图片的类别概率分布。模型的测试部分暂不讨论。

手写数字图片 MNIST 数据集的训练误差曲线如图 3.10 所示，由于 3 层的神经网络表达能力较强，手写数字图片识别任务相对简单，
误差值可以较快速、稳定地下降，其中， 把对数据集的所有样本迭代一遍叫作一个 Epoch，我们可以在间隔数个 Epoch 后测试模型
的准确率等指标，方便监控模型的训练效果。
"""