# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 15:31
"""
自 2012 年 AlexNet [3]的提出以来，各种各样的深度卷积神经网络模型相继被提出，其中比较有代表性的有 VGG 系列 [8]，
GoogLeNet 系列 [9]，ResNet 系列 [10]，DenseNet系列 [11]等，他们的网络层数整体趋势逐渐增多。
以网络模型在 ILSVRC 挑战赛 ImageNet数据集上面的分类性能表现为例，如图 10.42 所示，在 AlexNet 出现之前的网络模型都是浅层的神经网络，
Top-5 错误率均在 25%以上，AlexNet 8 层的深层神经网络将 Top-5 错误率降低至 16.4%，性能提升巨大，后续的 VGG、GoogleNet 模型继续将错误率降低至6.7%；
ResNet 的出现首次将网络层数提升至 152 层，错误率也降低至 3.57%。

10.9.1AlexNet
2012 年，ILSVRC12 挑战赛 ImageNet 数据集分类任务的冠军 Alex Krizhevsky 提出了 8层的深度神经网络模型 AlexNet，
它接收输入为22 × 22 大小的彩色图片数据，经过五个卷积层和三个全连接层后得到样本属于 1000 个类别的概率分布。
为了降低特征图的维度，AlexNet 在第 1、2、5 个卷积层后添加了 Max Pooling 层，如图 10.43 所示，网络的参数量达到了 6000 万个。
为了能够在当时的显卡设备 NVIDIA GTX 580(3GB 显存)上训练模型，Alex Krizhevsky 将卷积层、前 2 个全连接层等拆开在两块显卡上面分别训练，
最后一层合并到一张显卡上面，进行反向传播更新。AlexNet 在 ImageNet 取得了 15.3%的 Top-5 错误率，比第二名在错误率上降低了 10.9%。

AlexNet 的创新之处在于：
❑ 层数达到了较深的 8 层。
❑ 采用了 ReLU 激活函数，过去的神经网络大多采用 Sigmoid 激活函数，计算相对复杂，容易出现梯度弥散现象。
❑ 引入 Dropout 层。Dropout 提高了模型的泛化能力，防止过拟合。


10.9.2VGG系列
AlexNet 模型的优越性能启发了业界朝着更深层的网络模型方向研究。2014 年，ILSVRC14 挑战赛 ImageNet 分类任务的亚军牛津大学 VGG 实验室
提出了 VGG11、VGG13、VGG16、VGG19 等一系列的网络模型(图 10.45)，并将网络深度最高提升至 19 层 [8]。
以 VGG16 为例，它接受22 × 22 大小的彩色图片数据，经过 2 个 Conv-Conv- Pooling 单元，和 3 个 Conv-Conv-Conv-Pooling 单元的堆叠，
最后通过 3 层全连接层输出当前图片分别属于 1000 类别的概率分布，如图 10.44 所示。VGG16 在 ImageNet 取得了7.4%的 Top-5 错误率，
比 AlexNet 在错误率上降低了 7.9%。

VGG 系列网络的创新之处在于：
❑ 层数提升至 19 层。
❑ 全部采用更小的3 × 3卷积核，相对于 AlexNet 中7×7的卷积核，参数量更少，计算代价更低。
❑ 采用更小的池化层2 × 2窗口和步长𝑠 = 2，而 AlexNet 中是步长𝑠 = 2、3 × 3的池化窗口。


10.9.3GooLeNet
3 × 3的卷积核参数量更少，计算代价更低，同时在性能表现上甚至更优越，因此业界开始探索卷积核最小的情况：1×1卷积核。
如下图 10.46 所示，输入为 3 通道的5×5图片，与单个1×1的卷积核进行卷积运算，每个通道的数据与对应通道的卷积核运算，
得到3 个通道的中间矩阵，对应位置相加得到最终的输出张量。对于输入 shape 为 [b, ℎ, 𝑤, 𝑐𝑖𝑛] ， 1×1卷积层的输出为 [b, ℎ, 𝑤, 𝑐out] ，
其中𝑐𝑖𝑛为输入数据的通道数，𝑐𝑜𝑢𝑡为输出数据的通道数，也是1×1卷积核的数量。1×1卷积核的一个特别之处在于，
它可以不改变特征图的宽高，而只对通道数𝑐进行变换。

2014 年，ILSVRC14 挑战赛的冠军 Google 提出了大量采用3 × 3和 1× 1卷积核的网络模型：GoogLeNet，网络层数达到了 22 层 [9]。
虽然 GoogLeNet 的层数远大于 AlexNet，但是它的参数量却只有 AlexNet 的 1 / 12，同时性能也远好于 AlexNet。
在 ImageNet 数据集分类任务上，GoogLeNet 取得了 6.7%的 Top-5 错误率，比 VGG16 在错误率上降低了 0.7%。
GoogLeNet 网络采用模块化设计的思想，通过大量堆叠 Inception 模块，形成了复杂的网络结构。
如下图 10.47 所示，Inception 模块的输入为𝑿，通过 4 个子网络得到 4 个网络输出，在通道轴上面进行拼接合并，形成 Inception 模块的输出。
这 4 个子网络是：
❑ 1× 1卷积层。
❑ 1× 1卷积层，再通过一个3 × 3卷积层。
❑ 1× 1卷积层，再通过一个5 × 5卷积层。
❑ 3 × 3最大池化层，再通过1 × 1卷积层

GoogLeNet 的网络结构如图 10.48 所示，其中红色框中的网络结构即为图 10.47的网络结构。


"""

