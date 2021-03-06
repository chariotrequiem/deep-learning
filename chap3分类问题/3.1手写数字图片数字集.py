# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/13 16:41
"""
前面已经介绍了用于连续值预测的线性回归模型，现在我们来挑战分类问题。分类问题的一个典型应用就是教会机器
如何自动识别图片中物体的种类。考虑图片分类中最简单的任务之一：0~9 数字图片识别，它相对简单，而且也具有非常广泛的应用价值，
比如邮政编码、快递单号、手机号码等都属于数字图片识别范畴。我们将以数字图片识别为例，探索如何用机器学习的方法去解决这个问题。

3.1 手写数字图片数据集
机器学习需要从数据中间学习，因此首先需要采集大量的真实样本数据。以手写的数字图片识别为例，我们需要收集大量的
由真人书写的 0~9 的数字图片，为了便于存储和计算，一般把收集的原始图片缩放到某个固定的大小(Size 或 Shape)，比如
224个像素的行和224个像素的列(224 × 224)，或者96个像素的行和96个像素的列(96 × 96)，这张图片将作为输入数据x。
同时，我们需要给每一张图片标注一个标签(Label)，它将作为图片的真实值𝑦，这个标签表明这张图片属于哪一个具体的类别，
一般通过映射方式将类别名一一对应到从0开始编号的数字，比如说硬币的正反面，我们可以用0来表示硬币的反面，
用1来表示硬币的正面，当然也可以反过来1表示硬币的反面，这种编码方式叫作数字编码(Number Encoding)。
对于手写数字图片识别问题，编码更为直观，我们用数字的0~9来表示类别名字为0~9的图片。

如果希望模型能够在新样本上也能具有良好的表现，即模型泛化能力(Generalization Ability)较好，
那么我们应该尽可能多地增加数据集的规模和多样性(Variance)，使得我们用于学习的训练数据集与真实的手写数字图片的分布
(Ground-truth Distribution)尽可能的逼近，这样在训练数据集上面学到了模型能够很好的用于未见过的手写数字图片的预测。

为了方便业界统一测试和评估算法，发布了手写数字图片数据集，命名为 MNIST，它包含了 0~9 共10种数字的手写图片，
每种数字一共有7000张图片，采集自不同书写风格的真实手写图片，一共70000张图片。其中60000张图片作为训练集𝔻train(Training Set)，
用来训练模型，剩下10000张图片作为测试集𝔻test(Test Set)，用来预测或者测试，训练集和测试集共同组成了整个 MNIST 数据集。

考虑到手写数字图片包含的信息比较简单，每张图片均被缩放到28 × 28的大小，同时只保留了灰度信息。这些图片由真人书写，
包含了如字体大小、书写风格、粗细等丰富的样式，确保这些图片的分布与真实的手写数字图片的分布尽可能的接近，从而保证了模型的泛化能力。

现在我们来看下图片的表示方法。一张图片包含了ℎ行(Height/Row)，𝑤列 (Width/Column)，每个位置保存了像素(Pixel)值，
像素值一般使用 0~255 的整形数值来表达颜色强度信息，例如 0 表示强度最低，255 表示强度最高。如果是彩色图片，则每个像素
点包含了 R、G、B 三个通道的强度信息，分别代表红色通道、绿色通道、蓝色通道的颜色强度，所以与灰度图片不同，
它的每个像素点使用一个 1 维、长度为 3 的向量(Vector)来表示，向量的 3 个元素依次代表了当前像素点上面的 R、G、B 颜色强值，因此彩色图片
需要保存为形状是[ℎ, 𝑤, 3]的张量(Tensor，可以通俗地理解为 3 维数组)。
如果是灰度图片，则使用一个数值来表示灰度强度，例如 0 表示纯黑，255 表示纯白，因此它只需要一个形状为[ℎ, 𝑤]的二维矩阵(Matrix)
来表示一张图片信息(也可以保存为[ℎ, 𝑤, 1]形状的张量)。图 3.3 演示了内容为 8 的数字图片的矩阵内容，可以看到，图片中黑色的像素用 0 表
示，灰度信息用 0~255 表示，图片中越白的像素点，对应矩阵位置中数值也就越大。

目前常用的深度学习框架，如 TensorFlow、PyTorch 等，都可以非常方便地通过数行代码自动下载、管理和加载 MNIST 数据集，
不需要我们额外编写代码，使用起来非常方便。我们这里利用 TensorFlow 自动在线下载 MNIST 数据集，并转换为 Numpy 数组格式。

load_data()函数返回两个元组(tuple)对象，第一个是训练集，第二个是测试集，每个 tuple的第一个元素是多个训练图片数据𝑿，
第二个元素是训练图片对应的类别数字𝒀。其中训练集𝑿的大小为(60000,28,28)，代表了60000个样本，每个样本由 28 行、28 列构成，
由于是灰度图片，故没有 RGB 通道；训练集𝒀的大小为(60000)，代表了这 60000 个样本的标签数字，
每个样本标签用一个范围为 0~9 的数字表示。测试集 X 的大小为(10000,28,28)，代表了 10000 张测试图片，Y 的大小为(10000)。

从 TensorFlow 中加载的 MNIST 数据图片，数值的范围为[0,255]。在机器学习中间，一般希望数据的范围在0周围的小范围内分布。
通过预处理步骤，我们把[0,255]像素范围归一化(Normalize)到[0,1.]区间，再缩放到[−1,1]区间，从而有利于模型的训练。

每一张图片的计算流程是通用的，我们在计算的过程中可以一次进行多张图片的计算，充分利用 CPU 或 GPU 的并行计算能力。
我们用形状为[ℎ, 𝑤]的矩阵来表示一张图片，对于多张图片来说，我们在前面添加一个数量维度(Dimension)，
使用形状为[𝑏, ℎ, 𝑤]的张量来表示，其中𝑏代表了批量(Batch Size)；多张彩色图片可以使用形状为[𝑏, ℎ, 𝑤, 𝑐]的张量来
表示，其中𝑐表示通道数量(Channel)，彩色图片𝑐 = 3。通过 TensorFlow 的 Dataset 对象可以方便完成模型的批量训练，
只需要调用 batch()函数即可构建带 batch 功能的数据集对象。
"""
import os
import tensorflow as tf  # 导入TF库
from tensorflow import keras  # 导入TF子库keras
from tensorflow.keras import layers, optimizers, datasets  # 导入TF子库等
(x, y), (x_val, y_val) = datasets.mnist.load_data()  # 加载MNIST数据集
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1  # 转换成浮点张量，并缩放到-1~1
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为整形张量
y = tf.one_hot(y, depth=10)  # one-hot编码
print(x.shape, y.shape)
# print(x)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建数据集对象
train_dataset = train_dataset.batch(512)  # 批量训练
