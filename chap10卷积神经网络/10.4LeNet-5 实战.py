# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 21:13
"""
1990 年代，Yann LeCun 等人提出了用于手写数字和机器打印字符图片识别的神经网络，被命名为 LeNet-5 [4]。LeNet-5 的提出，
使得卷积神经网络在当时能够成功被商用，广泛应用在邮政编码、支票号码识别等任务中。下图 10.30 是 LeNet-5 的网络结构图，
它接受32 × 32大小的数字、字符图片，经过第一个卷积层得到 28 28 形状的张量，经过一个向下采样层，张量尺寸缩小到 ，
经过第二个卷积层，得到 形状的张量，同样经过下采样层，张量尺寸缩小到 ，在进入全连接层之前，先将张量打成[b, 400]的张量，
送入输出节点数分别为 120、84 的 2 个全连接层，得到 [b, 84] 的张量，最后通过 Gaussian connections 层。

现在看来，LeNet-5 网络层数较少(2 个卷积层和 2 个全连接层)，参数量较少，计算代价较低，尤其在现代 GPU 的加持下，数分钟即可训练好 LeNet-5 网络。

我们在 LeNet-5 的基础上进行了少许调整，使得它更容易在现代深度学习框架上实现。首先我们将输入𝑿形状由32 × 32调整为28 × 28，
然后将 2 个下采样层实现为最大池化层(降低特征图的高、宽，后续会介绍)，最后利用全连接层替换掉 Gaussian connections层。
下文统一称修改的网络也为 LeNet-5 网络。网络结构图如图 10.31 所示。

我们基于 MNIST 手写数字图片数据集训练 LeNet-5 网络，并测试其最终准确度。前面
已经介绍了如何在 TensorFlow 中加载 MNIST 数据集，此处不再赘述。
首先通过 Sequential 容器创建 LeNet-5，代码如下：
from tensorflow.keras import Sequential
network = Sequential([ # 网络容器
layers.Conv2D(6,kernel_size=3,strides=1), # 第一个卷积层, 6 个 3x3 卷积核
layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
layers.ReLU(), # 激活函数
layers.Conv2D(16,kernel_size=3,strides=1), # 第二个卷积层, 16 个 3x3 卷积核
layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
layers.ReLU(), # 激活函数
layers.Flatten(), # 打平层，方便全连接层处理
layers.Dense(120, activation='relu'), # 全连接层，120 个节点
layers.Dense(84, activation='relu'), # 全连接层，84 节点
layers.Dense(10) # 全连接层，10 个节点
])
# build 一次网络模型，给输入 X 的形状，其中 4 为随意给的 batchsz
"""