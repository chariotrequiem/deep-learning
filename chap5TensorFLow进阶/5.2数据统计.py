# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 11:03
"""
在神经网络的计算过程中，经常需要统计数据的各种属性，如最值、最值位置、均值、范数等信息。由于张量通常较大，
直接观察数据很难获得有用信息，通过获取这些张量的统计信息可以较轻松地推测张量数值的分布。

5.2.1向量范数
向量范数(Vector Norm)是表征向量"长度"的一种度量方法，他可以推广到张量上。
在神经网络中，常用来表示张量的权值大小，梯度大小等。常用的向量范数有：
❑ L1 范数，定义为向量𝒙的所有元素绝对值之和
          ‖𝒙‖1 = ∑𝑖 |𝑥𝑖|
❑ L2 范数，定义为向量𝒙的所有元素的平方和，再开根号
         ‖𝒙‖2 = √∑𝑖 |𝑥𝑖|2
❑ ∞ −范数，定义为向量𝒙的所有元素绝对值的最大值：
         ‖𝒙‖∞ = 𝑚𝑎𝑥𝑖(|𝑥𝑖|)
对于矩阵和张量，同样可以利用向量范数的计算公式，等价于将矩阵和张量打平成向量后计算。
在 TensorFlow 中，可以通过 tf.norm(x, ord)求解张量的 L1、L2、∞等范数，其中参数ord 指定为 1、2 时计算 L1、L2 范数，
指定为 np.inf 时计算∞ −范数，例如：
In [13]: x = tf.ones([2,2])
tf.norm(x,ord=1) # 计算 L1 范数
Out[13]: <tf.Tensor: id=183, shape=(), dtype=float32, numpy=4.0>
In [14]: tf.norm(x,ord=2) # 计算 L2 范数
Out[14]: <tf.Tensor: id=189, shape=(), dtype=float32, numpy=2.0>
In [15]: import numpy as np
tf.norm(x,ord=np.inf) # 计算∞范数
Out[15]: <tf.Tensor: id=194, shape=(), dtype=float32, numpy=1.0>



5.2.2最值、均值、和
通过 tf.reduce_max、tf.reduce_min、tf.reduce_mean、tf.reduce_sum 函数可以求解张量在某个维度上的最大、最小、均值、和，
也可以求全局最大、最小、均值、和信息。

考虑 shape 为[4,10]的张量，其中，第一个维度代表样本数量，第二个维度代表了当前样本分别属于 10 个类别的概率，
需要求出每个样本的概率最大值为，可以通过tf.reduce_max 函数实现：
In [16]: x = tf.random.normal([4,10]) # 模型生成概率
tf.reduce_max(x,axis=1) # 统计概率维度上的最大值
Out[16]:<tf.Tensor: id=203, shape=(4,), dtype=float32,
numpy=array([1.2410722 , 0.88495886, 1.4170984 , 0.9550192 ], dtype=float32)>

返回长度为 4 的向量，分别代表了每个样本的最大概率值。同样求出每个样本概率的最小
值，实现如下：
In [17]: tf.reduce_min(x,axis=1) # 统计概率维度上的最小值
Out[17]:<tf.Tensor: id=206, shape=(4,), dtype=float32, numpy=array([-
0.27862206, -2.4480672 , -1.9983795 , -1.5287997 ], dtype=float32)>

求出每个样本的概率的均值，实现如下：
In [18]: tf.reduce_mean(x,axis=1) # 统计概率维度上的均值
Out[18]:<tf.Tensor: id=209, shape=(4,), dtype=float32,
numpy=array([ 0.39526337, -0.17684573, -0.148988 , -0.43544054], dtype=float32)>

当不指定 axis 参数时，tf.reduce_*函数会求解出全局元素的最大、最小、均值、和等数据，例如：
In [19]:x = tf.random.normal([4,10])
# 统计全局的最大、最小、均值、和，返回的张量均为标量
tf.reduce_max(x),tf.reduce_min(x),tf.reduce_mean(x)
Out [19]: (<tf.Tensor: id=218, shape=(), dtype=float32, numpy=1.8653786>,
<tf.Tensor: id=220, shape=(), dtype=float32, numpy=-1.9751656>,
<tf.Tensor: id=222, shape=(), dtype=float32, numpy=0.014772797>)

在求解误差函数时，通过 TensorFlow 的 MSE 误差函数可以求得每个样本的误差，需要计算样本的平均误差，
此时可以通过 tf.reduce_mean 在样本数维度上计算均值，实现如下：
In [20]:
out = tf.random.normal([4,10]) # 模拟网络预测输出
y = tf.constant([1,2,2,0]) # 模拟真实标签
y = tf.one_hot(y,depth=10) # one-hot 编码
loss = keras.losses.mse(y,out) # 计算每个样本的误差
loss = tf.reduce_mean(loss) # 平均误差，在样本数维度上取均值
loss # 误差标量
Out[20]:
<tf.Tensor: id=241, shape=(), dtype=float32, numpy=1.1921183>

与均值函数相似的是求和函数 tf.reduce_sum(x, axis)，它可以求解张量在 axis 轴上所有特征的和：
In [21]:out = tf.random.normal([4,10])
tf.reduce_sum(out,axis=-1) # 求最后一个维度的和
Out[21]:<tf.Tensor: id=303, shape=(4,), dtype=float32, numpy=array([-
0.588144 , 2.2382064, 2.1582587, 4.962141 ], dtype=float32)>


除了希望获取张量的最值信息，还希望获得最值所在的位置索引号，例如分类任务的标签预测，就需要知道概率最大值所在的位置索引号，
一般把这个位置索引号作为预测类别。考虑 10 分类问题，我们得到神经网络的输出张量 out，shape 为[2,10]，
代表了 2 个样本属于 10 个类别的概率，由于元素的位置索引代表了当前样本属于此类别的概率，
预测时往往会选择概率值最大的元素所在的索引号作为样本类别的预测值，例如：
In [22]:out = tf.random.normal([2,10])
out = tf.nn.softmax(out, axis=1) # 通过 softmax 函数转换为概率值
out
Out[22]:<tf.Tensor: id=257, shape=(2, 10), dtype=float32, numpy=
array([[0.18773547, 0.1510464 , 0.09431915, 0.13652141, 0.06579739,
 0.02033597, 0.06067333, 0.0666793 , 0.14594753, 0.07094406],
 [0.5092072 , 0.03887136, 0.0390687 , 0.01911005, 0.03850609,
 0.03442522, 0.08060656, 0.10171875, 0.08244187, 0.05604421]],
 dtype=float32)>

以第一个样本为例，可以看到，它概率最大的索引为𝑖 = 0，最大概率值为 0.1877。
由于每个索引号上的概率值代表了样本属于此索引号的类别的概率，因此第一个样本属于 0 类的概率最大，
在预测时考虑第一个样本应该最有可能属于类别 0。这就是需要求解最大值的索引号的一个典型应用。
通过 tf.argmax(x, axis)和 tf.argmin(x, axis)可以求解在 axis 轴上，x 的最大值、最小值所在的索引号，例如：
In [23]:pred = tf.argmax(out, axis=1) # 选取概率最大的位置
pred
Out[23]:<tf.Tensor: id=262, shape=(2,), dtype=int64, numpy=array([0, 0], dtype=int64)>
可以看到，这 2 个样本概率最大值都出现在索引 0 上，因此最有可能都是类别 0，我们可 以将类别 0 作为这 2 个样本的预测类别。
"""