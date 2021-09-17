# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 16:21
"""
全连接层是神经网络最基本的网络类型，对后续神经网络类型的研究有巨大的贡献，全连接层前向计算流程相对简单，梯度求导也较简单，
但是它有一个最大的缺陷，在处理较大特征长度的数据时，全连接层的参数量往往较大，使得深层数的全连接网络参数量巨大，
训练起来比较困难。近年来，社交媒体的发达产生了海量的图片、视频、文本等数字资源，极大地促进了神经网络在计算机视觉、自然语言处理等领域中的研究，
相继提出了一系列的神经网络变种类型。

6.7.1卷积神经网络
如何识别、分析并理解图片、视频等数据是计算机视觉的一个核心问题，全连接层在处理高维度的图片、视频数据时往往出现网络参数量巨大，
训练非常困难的问题。通过利用局部相关性和权值共享的思想，Yann Lecun 在 1986 年提出了卷积神经网络(Convolutional Neural Network，简称 CNN)。
随着深度学习的兴盛，卷积神经网络在计算机视觉中的表现大大地超越了其它算法模型，呈现统治计算机视觉领域之势。
这其中比较流行的模型有用于图片分类的 AlexNet、VGG、GoogLeNet、ResNet、DenseNet 等，
用于目标识别的 RCNN、Fast RCNN、Faster RCNN、Mask RCNN、YOLO、SSD 等。
我们将在第10 章详细介绍卷积神经网络原理。


6.7.2循环神经网络
除了具有空间结构的图片、视频等数据外，序列信号也是非常常见的一种数据类型，其中一个最具代表性的序列信号就是文本数据。
如何处理并理解文本数据是自然语言处理的一个核心问题。卷积神经网络由于缺乏 Memory 机制和处理不定长序列信号的能力，
并不擅长序列信号的任务。循环神经网络(Recurrent Neural Network，简称 RNN)在 Yoshua Bengio、Jürgen Schmidhuber 等人的持续研究下，
被证明非常擅长处理序列信号。1997年，Jürgen Schmidhuber 提出了 LSTM 网络，作为 RNN 的变种，
它较好地克服了 RNN 缺乏长期记忆、不擅长处理长序列的问题，在自然语言处理中得到了广泛的应用。
基于LSTM 模型，Google 提出了用于机器翻译的 Seq2Seq 模型，并成功商用于谷歌神经机器翻译系统(GNMT)。
其他的 RNN 变种还有 GRU、双向 RNN 等。我们将在第 11 章详细介绍循环神经网络原理。


6.7.3注意力(机制)网络
RNN 并不是自然语言处理的最终解决方案，近年来随着注意力机制(Attention Mechanism)的提出，克服了 RNN 训练不稳定、难以并行化等缺陷，
在自然语言处理和图片生成等领域中逐渐崭露头角。注意力机制最初在图片分类任务上提出，但逐渐开始侵蚀NLP 各大任务。
2017 年，Google 提出了第一个利用纯注意力机制实现的网络模型Transformer，
随后基于 Transformer 模型相继提出了一系列的用于机器翻译的注意力网络模型，如 GPT、BERT、GPT-2 等。在其它领域，基于注意力机制，
尤其是自注意力(Self- Attention)机制构建的网络也取得了不错的效果，比如基于自注意力机制的 BigGAN 模型等。


6.7.4图卷积神经网络
图片、文本等数据具有规则的空间、时间结构，称为 Euclidean Data(欧几里德数据)。
卷积神经网络和循环神经网络被证明非常擅长处理这种类型的数据。而像类似于社交网络、通信网络、蛋白质分子结构等一系列的不规则空间拓扑结构的数据，
它们显得力不从心。2016 年，Thomas Kipf 等人基于前人在一阶近似的谱卷积算法上提出了图卷积网络(Graph Convolution Network，GCN)模型。
GCN 算法实现简单，从空间一阶邻居信息聚合的角度也能直观地理解，在半监督任务上取得了不错效果。随后，一系列的网络模型相继被提出，
如 GAT，EdgeConv，DeepGCN 等。
"""