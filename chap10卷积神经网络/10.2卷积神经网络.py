# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 19:32
"""
卷积神经网络通过充分利用局部相关性和权值共享的思想，大大地减少了网络的参数量，从而提高训练效率，更容易实现超大规模的深层网络。
2012 年，加拿大多伦多大学Alex Krizhevsky 将深层卷积神经网络应用在大规模图片识别挑战赛 ILSVRC-2012 上，
在ImageNet 数据集上取得了 15.3% 的 Top-5 错误率，排名第一，相对于第二名在 Top-5 错误率上降低了 10.9% [3]，
这一巨大突破引起了业界强烈关注，卷积神经网络迅速成为计算机视觉领域的新宠，随后在一系列的任务中，基于卷积神经网络的形形色色的模型相继被提
出，并在原有的性能上取得了巨大提升。

现在我们来介绍卷积神经网络层的具体计算流程。以 2D 图片数据为例，卷积层接受高、宽分别为ℎ、𝑤，通道数为𝑐𝑖𝑛的输入特征图𝑿，
在𝑐𝑜𝑢𝑡个高、宽都为𝑘，通道数为𝑐𝑖𝑛的卷积核作用下，生成高、宽分别为ℎ′、𝑤′，通道数为𝑐𝑜𝑢𝑡的特征图输出。
需要注意的是，卷积核的高宽可以不等，为了简化讨论，这里仅讨论高宽都为𝑘的情况，之后可以轻松推广到高、宽不等的情况。

我们首先从单通道输入、单卷积核的情况开始讨论，然后推广至多通道输入、单卷积核，最后讨论最常用，也是最复杂的多通道输入、多个卷积核的卷积层实现。

10.2.1单通道输入和单卷积核
首先讨论单通道输入𝑐𝑖𝑛 = ，如灰度图片只有灰度值一个通道，单个卷积核𝑐𝑜𝑢𝑡 = 的情况。以输入𝑿为 × 的矩阵，卷积核为3 × 3的矩阵为例，
如下图 10.12 所示。与卷积核同大小的感受野(输入𝑿上方的绿色方框)首先移动至输入𝑿最左上方，选中输入𝑿上3 × 3 的感受野元素，
与卷积核(图片中间3 × 3方框)对应元素相乘：
⨀符号表示哈达马积(Hadamard Product)，即矩阵的对应元素相乘，它与矩阵相乘符号@是矩阵的二种最为常见的运算形式。
运算后得到3 × 3的矩阵，这 9 个数值全部相加： -1-1+0-1+2+6+0-2+4 = 7
得到标量 7，写入输出矩阵第一行、第一列的位置，如图 10.12 所示。
完成第一个感受野区域的特征提取后，感受野窗口向右移动一个步长单位(Strides，记 为𝑠，默认为 1)，
选中图 10.13 中绿色方框中的 9 个感受野元素，按照同样的计算方法，与卷积核对应元素相乘累加，得到输出 10，写入第一行、第二列位置。

感受野窗口再次向右移动一个步长单位，选中图 10.14 中绿色方框中的元素，并与卷积核相乘累加，得到输出 3，并写入输出的第一行、第三列位置，如图 10.14 所示。

此时感受野已经移动至输入𝑿的有效像素的最右边，无法向右边继续移动(在不填充无效元素的情况下），因此感受野窗口向下移动一个步长单位(𝑠 = 1)，
并回到当前行的行首位置，继续选中新的感受野元素区域，如图 10.15 所示，与卷积核运算得到输出-1。此时的感受野由于经过向下移动一个步长单位，
因此输出值-1 写入第二行、第一列位置。

按照上述方法，每次感受野向右移动𝑠 = 个步长单位，若超出输入边界，则向下移动𝑠 = 1个步长单位，并回到行首，直到感受野移动至最右边、最下方位置，如下图 10.16 所示。
每次选中的感受野区域元素，和卷积核对应元素相乘累加，并写入输出的对应位置。最终输出我们得到一个3 × 3的矩阵，比输入5×5略小，这是因为感受野不能超出元素边界
的缘故。可以观察到，卷积运算的输出矩阵大小由卷积核的大小𝑘，输入𝑿的高宽ℎ/𝑤，移动步长𝑠，是否填充等因素共同决定。
这里为了演示计算过程，预绘制了一个与输入等大小的网格，并不表示输出高宽为5×5，这里的实际输出高宽只有3 × 3。


10.2.2多通道输入和单核卷积
多通道输入的卷积层更为常见，比如彩色的图片包含了 R/G/B 三个通道，每个通道上面的像素值表示 R/G/B 色彩的强度。
下面我们以 3 通道输入、单个卷积核为例，将单通道输入的卷积运算方法推广到多通道的情况。
如图 10.17 中所示，每行的最左边5×5的矩阵表示输入𝑿的 1~3 通道，第 2 列的3 × 3矩阵分别表示卷积核的 1~3 通道，
第 3 列的矩阵表示当前通道上运算结果的中间矩阵，最右边一个矩阵表示卷积层运算的最终输出。

在多通道输入的情况下，卷积核的通道数需要和输入𝑿的通道数量相匹配，卷积核的第𝑖个通道和𝑿的第𝑖个通道运算，得到第𝑖个中间矩阵，
此时可以视为单通道输入与单卷积核的情况，所有通道的中间矩阵对应元素再次相加，作为最终输出。

具体的计算流程如下：在初始状态，如图 10.17 所示，每个通道上面的感受野窗口同步落在对应通道上面的最左边、最上方位置，
每个通道上感受野区域元素与卷积核对应通道上面的矩阵相乘累加，分别得到三个通道上面的输出 7、-11、-1 的中间变量，
这些中间变量相加得到输出-5，写入对应位置。

随后，感受野窗口同步在𝑿的每个通道上向右移动𝑠 = 个步长单位，此时感受野区域元素如下图 10.18 所示，
每个通道上面的感受野与卷积核对应通道上面的矩阵相乘累加，得到中间变量 10、20、20，全部相加得到输出 50，写入第一行、第二列元素位置。

以此方式同步移动感受野窗口，直至最右边、最下方位置，此时全部完成输入和卷积核的卷积运算，得到3 × 3的输出矩阵，如图 10.19 所示。

整个的计算示意图如下图 10.20 所示，输入的每个通道处的感受野均与卷积核的对应通道相乘累加，得到与通道数量相等的中间变量，
这些中间变量全部相加即得到当前位置的输出值。输入通道的通道数量决定了卷积核的通道数。
一个卷积核只能得到一个输出矩阵，无论输入𝑿的通道数量。

一般来说，一个卷积核只能完成某种逻辑的特征提取，当需要同时提取多种逻辑特征时，可以通过增加多个卷积核来得到多种特征，
提高神经网络的表达能力，这就是多通道输入、多卷积核的情况。


10.2.3多通道输入、多卷积核
多通道输入、多卷积核是卷积神经网络中最为常见的形式，前面我们已经介绍了单卷积核的运算过程，每个卷积核和输入𝑿做卷积运算，
得到一个输出矩阵。当出现多卷积核时，第𝑖 (𝑖 ∈ 𝑛 ，𝑛为卷积核个数)个卷积核与输入𝑿运算得到第𝑖个输出矩阵(也称为输出张量𝑶的通道𝑖），
最后全部的输出矩阵在通道维度上进行拼接(Stack 操作，创建输出通道数的新维度)，产生输出张量𝑶，𝑶包含了𝑛个通道数。

以 3 通道输入、2 个卷积核的卷积层为例。第一个卷积核与输入𝑿运算得到输出𝑶的第一个通道，第二个卷积核与输入𝑿运算得到输出𝑶的第二个通道，
如下图 10.21 所示，输出的两个通道拼接在一起形成了最终输出𝑶。每个卷积核的大小𝑘、步长𝑠、填充设定等都是统一设置，
这样才能保证输出的每个通道大小一致，从而满足拼接的条件。


10.2.4步长
在卷积运算中，如何控制感受野布置的密度呢？对于信息密度较大的输入，如物体数量很多的图片，为了尽可能的少漏掉有用信息，
在网络设计的时候希望能够较密集地布置感受野窗口；对于信息密度较小的输入，比如全是海洋的图片，可以适量的减少感受野窗口的数量。
感受野密度的控制手段一般是通过移动步长(Strides)实现的。

步长是指感受野窗口每次移动的长度单位，对于 2D 输入来说，分为沿𝑥(向右)方向和𝑦(向下)方向的移动长度。为了简化讨论，
这里只考虑𝑥/𝑦方向移动步长相同的情况，这也是神经网络中最常见的设定。如下图 10.22 所示，绿色实线代表的感受野窗口的位置是当前位置，
绿色虚线代表是上一次感受野所在位置，从上一次位置移动到当前位置的移动长度即是步长的定义。图 10.22 中感受野沿𝑥方向的步长为 2，表达为步长𝑠 = 2。

当感受野移动至输入𝑿右边的边界时，感受野向下移动一个步长𝑠 = 2，并回到行首。如下图 10.23 所示，感受野向下移动 2 个单位，并回到行首位置，进行相乘累加运算。

循环往复移动，直至达到最下方、最右边边缘位置，如图 10.24 所示，最终卷积层输出的高宽只有2 × 2。对比前面𝑠 = 的情形，
输出高宽由3 × 3降低为2 × 2，感受野的数量减少为仅 4 个。

可以看到，通过设定步长𝑠，可以有效地控制信息密度的提取。当步长设计的较小时，感受野以较小幅度移动窗口，有利于提取到更多的特征信息，
输出张量的尺寸也更大；当步长设计的较大时，感受野以较大幅度移动窗口，有利于减少计算代价，过滤冗余信息，输出张量的尺寸也更小。


10.2.5填充
经过卷积运算后的输出𝑶的高宽一般会小于输入𝑿的高宽，即使是步长𝑠 = 时，输出𝑶的高宽也会略小于输入𝑿高宽。在网络模型设计时，
有时希望输出𝑶的高宽能够与输入𝑿的高宽相同，从而方便网络参数的设计、残差连接等。为了让输出𝑶的高宽能够与输入𝑿的相等，
一般通过在原输入𝑿的高和宽维度上面进行填充(Padding)若干无效元素操作，得到增大的输入𝑿′。通过精心设计填充单元的数量，
在𝑿′上面进行卷积运算得到输出𝑶的高宽可以和原输入𝑿相等，甚至更大。

如下图 10.25 所示，在高/行方向的上(Top)、下(Bottom)方向，宽/列方向的左(Left)、 右(Right)均可以进行不定数量的填充操作，
填充的数值一般默认为 0，也可以填充自定义的数据。图 10.25 中上、下方向各填充 1 行，左、右方向各填充 2 列，得到新的输入𝑿′。

那么添加填充后的卷积层怎么运算呢？同样的方法，仅仅是把参与运算的输入从𝑿换成了填充后得到的新张量𝑿′。
如下图 10.26 所示，感受野的初始位置在填充后的𝑿′的左上方，完成相乘累加运算，得到输出 1，写入输出张量的对应位置。
移动步长𝑠 = 个单位，重复运算逻辑，得到输出 0，如图 10.27 所示。
循环往复，最终得到 × 的输出张量，如图 10.28 所示

通过精心设计的 Padding 方案，即上下左右各填充一个单位，记为𝑝 = 1，我们可以得到输出𝑶和输入𝑿的高、宽相等的结果；
在不加 Padding 的情况下，如下图 10.29 所示，只能得到3 × 3的输出𝑶，略小于输入𝑿


卷积神经层的输出尺寸 ℎ′ 𝑤′ 𝑐𝑜𝑢𝑡 由卷积核的数量𝑐𝑜𝑢𝑡，卷积核的大小𝑘，步长𝑠，填充数𝑝(只考虑上下填充数量𝑝ℎ相同，
左右填充数量𝑝𝑤相同的情况)以及输入𝑿的高宽ℎ/𝑤共同决定，它们之间的数学关系可以表达为：
                            ℎ′ = [(ℎ + 2 ∙𝑝ℎ − 𝑘)/s] + 1
                            𝑤′ = [(𝑤 + 2 ∙ 𝑝𝑤 − 𝑘) / 𝑠] + 1
其中𝑝ℎ、𝑝𝑤分别表示高、宽方向的填充数量，⌊∙⌋表示向下取整

在 TensorFlow 中，在𝑠 = 时，如果希望输出𝑶和输入𝑿高、宽相等，只需要简单地设置参数 padding=”SAME”即可使 TensorFlow 自动计算 padding 数量，非常方便。
"""
