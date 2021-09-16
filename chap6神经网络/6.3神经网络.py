# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/16 16:22
"""
通过层层堆叠图 6.4 中的全连接层，保证前一层的输出节点数与当前层的输入节点数匹配，即可堆叠出任意层数的网络。
我们把这种由神经元相互连接而成的网络叫做神经网络。如图 6.5 所示，通过堆叠 4 个全连接层，可以获得层数为 4 的神经网络，
由于每层均为全连接层，称为全连接网络。其中第 1~3 个全连接层在网络中间，称之为隐藏层 1、2、 3，最后一个全连接层的输出作为网络的输出，
称为输出层。隐藏层 1、2、3 的输出节点数分别为[256,128,64]，输出层的输出节点数为 10。

在设计全连接网络时，网络的结构配置等超参数可以按着经验法则自由设置，只需要遵循少量的约束即可。
例如，隐藏层 1 的输入节点数需和数据的实际特征长度匹配，每层的输入层节点数与上一层输出节点数匹配，
输出层的激活函数和节点数需要根据任务的具体设定进行设计。总的来说，神经网络模型的结构设计自由度较大，
如图 6.5 层中每层的输出节点数不一定要设计为[256,128,64,10]，可以自由搭配，如[256,256,64,10]或 [512,64,32,10]等都是可行的。
至于与哪一组超参数是最优的，这需要很多的领域经验知识和大量的实验尝试，或者可以通过 AutoML 技术搜索出较优设定。

6.3.1张量的实现方式
对于多层神经网络，以图 6.5 网络结构为例，需要分别定义各层的权值矩阵𝑾和偏置向量𝒃。有多少个全连接层，
则需要相应地定义数量相当的𝑾和𝒃，并且每层的参数只能用于对应的层，不能混淆使用。图 6.5 的网络模型实现如下：
# 隐藏层 1 张量
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隐藏层 2 张量
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隐藏层 3 张量
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
在计算时，只需要按照网络层的顺序，将上一层的输出作为当前层的输入即可，重复直至最后一层，并将输出层的输出作为网络的输出，
代码如下：
with tf.GradientTape() as tape: # 梯度记录器
    # x: [b, 28*28]
    # 隐藏层 1 前向计算，[b, 28*28] => [b, 256]
    h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    # 隐藏层 2 前向计算，[b, 256] => [b, 128]
    h2 = h1@w2 + b2
    h2 = tf.nn.relu(h2)
    # 隐藏层 3 前向计算，[b, 128] => [b, 64]
    h3 = h2@w3 + b3
    h3 = tf.nn.relu(h3)
    # 输出层前向计算，[b, 64] => [b, 10]
    h4 = h3@w4 + b4
最后一层是否需要添加激活函数通常视具体的任务而定，这里加不加都可以。

在使用 TensorFlow 自动求导功能计算梯度时，需要将前向计算过程放置在tf.GradientTape()环境中，
从而利用 GradientTape 对象的 gradient()方法自动求解参数的梯度，并利用 optimizers 对象更新参数。

6.3.2层方式实现
对于常规的网络层，通过层方式实现起来更加简洁高效。首先新建各个网络层类，并指定各层的激活函数类型：
# 导入常用网络层 layers
from tensorflow.keras import layers,Sequential
fc1 = layers.Dense(256, activation=tf.nn.relu) # 隐藏层 1
fc2 = layers.Dense(128, activation=tf.nn.relu) # 隐藏层 2
fc3 = layers.Dense(64, activation=tf.nn.relu) # 隐藏层 3
fc4 = layers.Dense(10, activation=None) # 输出层
在前向计算时，依序通过各个网络层即可，代码如下：
x = tf.random.normal([4,28*28])
h1 = fc1(x) # 通过隐藏层 1 得到输出
h2 = fc2(h1) # 通过隐藏层 2 得到输出
h3 = fc3(h2) # 通过隐藏层 3 得到输出
h4 = fc4(h3) # 通过输出层得到网络输出

对于这种数据依次向前传播的网络，也可以通过 Sequential 容器封装成一个网络大类对象，
调用大类的前向计算函数一次即可完成所有层的前向计算，使用起来更加方便，实现如下：
# 导入 Sequential 容器
from tensorflow.keras import layers,Sequential
# 通过 Sequential 容器封装为一个网络类
model = Sequential([
                   layers.Dense(256, activation=tf.nn.relu) , # 创建隐藏层 1
                   layers.Dense(128, activation=tf.nn.relu) , # 创建隐藏层 2
                   layers.Dense(64, activation=tf.nn.relu) , # 创建隐藏层 3
                   layers.Dense(10, activation=None) , # 创建输出层])
前向计算时只需要调用一次网络大类对象，即可完成所有层的按序计算：
out = model(x) # 前向计算得到输出

6.3.3优化目标
我们把神经网络从输入到输出的计算过程叫做前向传播(Forward Propagation)或前向计算。神经网络的前向传播过程，
也是数据张量(Tensor)从第一层流动(Flow)至输出层的过程，即从输入数据开始，途径每个隐藏层，直至得到输出并计算误差，
这也是 TensorFlow框架名字由来。

前向传播的最后一步就是完成误差的计算
ℒ = 𝑔(𝑓𝜃(𝒙),𝒚)
其中𝑓𝜃(∙)代表了利用𝜃参数化的神经网络模型，𝑔(∙)称之为误差函数，用来描述当前网络的
预测值𝑓𝜃(𝒙)与真实标签𝒚之间的差距度量，比如常用的均方差误差函数。ℒ称为网络的误差
(Error，或损失 Loss)，一般为标量。我们希望通过在训练集𝔻train上面学习到一组参数𝜃使 得训练的误差ℒ最小：
"""