# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/13 21:22
import tensorflow as tf
"""
TensorFlow 是一个面向深度学习算法的科学计算库，内部数据保存在张量(Tensor)对象上，
所有的运算操作(Operation，简称 OP)也都是基于张量对象进行的。复杂的神经网络算法本质上就是各种张量相乘、相加等基本运算操作的组合，
在深入学习深度学习算法之前，熟练掌握 TensorFlow 张量的基础操作方法十分重要。只有掌握了这些操作方法，
才能随心所欲地实现各种复杂新奇的网络模型，也才能深刻理解各种模型算法的本质。

4.1数据类型
首先介绍TensorFlow中的基本数据类型，包含数值类型、字符串类型和布尔类型
4.1.1数值类型
数值类型的张量是 TensorFlow 的主要数据载体，根据维度数来区分，可分为：
❑ 标量(Scalar)。单个的实数，如 1.2, 3.4 等，维度(Dimension)数为 0，shape 为[]。
❑ 向量(Vector)。𝑛个实数的有序集合，通过中括号包裹，如[1.2]，[1.2,3.4]等，维度数 为 1，长度不定，shape 为[𝑛]。
❑ 矩阵(Matrix)。𝑛行𝑚列实数的有序集合，如[[1,2],[3,4]]，也可以写成
                                [1 2
                                 3 4]
维度数为 2，每个维度上的长度不定，shape 为[𝑛, 𝑚]。
❑ 张量(Tensor)。所有维度数dim > 2的数组统称为张量。张量的每个维度也作轴(Axis)，一般维度代表了具体的物理含义，
比如 Shape 为[2,32,32,3]的张量共有 4 维，如果表示图片数据的话，每个维度/轴代表的含义分别是图片数量、图片高度、图片宽度、
图片通道数，其中 2 代表了 2 张图片，32 代表了高、宽均为 32，3 代表了 RGB 共 3 个通道。
张量的维度数以及每个维度所代表的具体物理含义需要由用户自行定义。
在 TensorFlow 中间，为了表达方便，一般把标量、向量、矩阵也统称为张量，不作区分，需要根据张量的维度数或形状自行判断，
本书也沿用此方式。
首先来看标量在 TensorFlow 是如何创建的，实现如下：
"""
a = 1.2  # python语言方式创建标量
aa = tf.constant(1.2)  # TF方式创建标量
print(type(a), type(aa), tf.is_tensor(aa))  # <class 'float'> <class 'tensorflow.python.framework.ops.EagerTensor'> True
"""
如果要使用 TensorFlow 提供的功能函数，须通过 TensorFlow 规定的方式去创建张量，而不能使用 Python 语言的标准变量创建方式。
通过print(x) 和x可以打印出张量x的相关信息
"""
x = tf.constant([1, 2., 3.3])  # tf.Tensor([1.  2.  3.3], shape=(3,), dtype=float32)
print(x)
"""
其中 id 是 TensorFlow 中内部索引对象的编号，shape 表示张量的形状，dtype 表示张量的数值精度，
张量 numpy()方法可以返回 Numpy.array 类型的数据，方便导出数据到系统的其他模块，代码如下。
"""
print(x.numpy())  # 将TF张量的数据导出为numpy数组格式
"""
与标量不同，向量的定义必须通过List传递给tf.constant()函数。例如，创建一个元素的向量：
In []:
a = tf.constant([1.2]) # 创建一个元素的向量
a, a.shape
Out[]:
(<tf.Tensor: id=8, shape=(1,), dtype=float32, numpy=array([1.2], 
dtype=float32)>,TensorShape([1]))
创建3个元素的向量
In [5]:
a = tf.constant([1,2, 3.]) # 创建 3 个元素的向量
a, a.shape
Out[5]:
(<tf.Tensor: id=11, shape=(3,), dtype=float32, numpy=array([1., 2., 3.], 
dtype=float32)>,
TensorShape([3]))
同样的方法，定义矩阵的实现如下：
a = tf.constant([[1, 2], [3, 4]])
a, a.shape
(<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
 array([[1, 2],
        [3, 4]])>,
 TensorShape([2, 2]))
 三维张量可以定义为：
a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
a, a.shape
(<tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
 array([[[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]]])>,
 TensorShape([2, 2, 2]))
 
4.1.2字符串类型
除了丰富的数值类型张量外，TensorFlow还支持字符串（String）类型的数据，例如在表示图片数据时，可以先记录图片的路径字符串，
再通过预处理函数根据路径读取图片张量。通过传入字符串对象即可创建字符串类型的张量，例如：
a = tf.constant('Hello, Deep Learning.')  # 创建字符串
a, a.shape
(<tf.Tensor: shape=(), dtype=string, numpy=b'Hello, Deep Learning.'>,
 TensorShape([]))
在tf.strings模块中，提供了常见的字符串类型的工具函数，如小写化lower()，拼接join(),长度length()、切分split()等。
例如，将字符串全部小写化实现为：
tf.strings.lower(a)
<tf.Tensor: shape=(), dtype=string, numpy=b'hello, deep learning.'>
深度学习算法主要还是以数值类型张量运算为主，字符串类型的数据使用频率较低，此处不做过多阐述

4.1.3布尔类型
为了方便表达比较运算的结果，TensorFLow还支持布尔类型(Boolean， 简称bool)的张量。
布尔类型的张量只需要传入python语言的布尔类型数据，抓换为TensorFlow内部布尔型即可，例如：
a = tf.constant(True)
<tf.Tensor: shape=(), dtype=bool, numpy=True>
同样的，创建布尔类型的向量，实现如下：
a = tf.constant([True, False]) # 创建布尔类型向量
<tf.Tensor: id=25, shape=(2,), dtype=bool, numpy=array([ True, False])>
需要注意的是，TensorFlow的布尔类型和Python语言的布尔类型并不等价，不能通用，例如：
a = tf.constant(True) # 创建 TF 布尔张量
a is True # TF 布尔类型张量与 python 布尔类型比较
a == True # 仅数值比较
Out[11]:
False # 对象不等价
<tf.Tensor: id=8, shape=(), dtype=bool, numpy=True> # 数值比较结果
"""