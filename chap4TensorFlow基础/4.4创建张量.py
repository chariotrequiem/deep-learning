# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/14 10:46
"""
在TensorFlow中，可以通过多种方式创建张量，如从Python列表对象创建，从Numpy数组创建，或者创建采样自某种已知分布的张量等。

4.4.1从数组、列表对象创建
Numpy Array数组和Python List列表是Python程序中间非常重要的数据载体容器，很多数据都是通过Python语言将数据加载至Array或者
List容器，再转换到tensor类型，通过TensoeFlow运算处理后导出到Array或者List容器，方便其他模块调用。
通过tf.convert_to_tensor函数可以创建新Tensor，并将保存在Python List对象或者Numpy Array对象中的数据导入到新Tensor中，例如：
In [22]:
tf.convert_to_tensor([1,2.]) # 从列表创建张量
Out[22]:
<tf.Tensor: id=86, shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>
In [23]:
tf.convert_to_tensor(np.array([[1,2.],[3,4]])) # 从数组中创建张量
Out[23]:
<tf.Tensor: id=88, shape=(2, 2), dtype=float64, numpy=array([[1., 2.], [3., 4.]])>
需要注意的是：Numpy浮点数数组默认使用64位精度保存数据，转换到Tensor类型时精度位tf.float64，可以在需要的时候将其转换为tf.float32类型
实际上，tf.constant()和tf.convert_to_tensor()都能自动的把Numpy数组或者Python列表数据实际上转化为Tensor类型，
这两个API命名来自TensorFLow1.x的命名习惯，在TensoeFlow2中函数的名字并不是很贴切，使用其一即可。

4.4.2创建全0或全1张量
将张量创建为全0或者全1数据是非常常见的张量初始化手段。考虑线性变换𝒚 = 𝑾𝒙 + 𝒃，将权值矩阵𝑾初始化为全1矩阵，
偏置𝒃初始化为全0向量，此时线性变化层输出y=x，因此是一种比较好的层初始化状态。通过tf.zeros()和tf.ones()即可创建任意形状，
且内容全0或全1的张量。例如，创建为0和为1的标量：
In [24]: tf.zeros([]),tf.ones([]) # 创建全 0，全 1 的标量
Out[24]:
(<tf.Tensor: id=90, shape=(), dtype=float32, numpy=0.0>,
<tf.Tensor: id=91, shape=(), dtype=float32, numpy=1.0>)

创建全 0 和全 1 的向量，实现如下：
In [25]: tf.zeros([1]),tf.ones([1]) # 创建全 0，全 1 的向量
Out[25]:
(<tf.Tensor: id=96, shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>,
<tf.Tensor: id=99, shape=(1,), dtype=float32, numpy=array([1.], dtype=float32)>)

创建全 0 的矩阵，例如：
In [26]: tf.zeros([2,2]) # 创建全 0 矩阵，指定 shape 为 2 行 2 列
Out[26]:
<tf.Tensor: id=104, shape=(2, 2), dtype=float32, numpy=array([[0., 0.],
 [0., 0.]], dtype=float32)>

创建全 1 的矩阵，例如：
In [27]: tf.ones([3,2]) # 创建全 1 矩阵，指定 shape 为 3 行 2 列
Out[27]:
<tf.Tensor: id=108, shape=(3, 2), dtype=float32, numpy=
array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)>

通过tf.zeros_like, tf.ones_like可以方便的新建与某个张量shape一致，且内容为全0或全1的张量。
例如，创建与张量A形状一样的全0张量：
In [28]: a = tf.ones([2,3]) # 创建一个矩阵
tf.zeros_like(a) # 创建一个与 a 形状相同，但是全 0 的新矩阵
Out[28]:
<tf.Tensor: id=113, shape=(2, 3), dtype=float32, numpy=
array([[0., 0., 0.],
      [0., 0., 0.]], dtype=float32)>
创建与张量𝑨形状一样的全 1 张量：
In [29]: a = tf.zeros([3,2]) # 创建一个矩阵
tf.ones_like(a) # 创建一个与 a 形状相同，但是全 1 的新矩阵
Out[29]:
<tf.Tensor: id=120, shape=(3, 2), dtype=float32, numpy=
array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)>

4.4.3创建自定义数值张量
除了初始化全为0，或全为1的张量之外，有时也需要全部初始化为某个自定义数值的张量，比如将张量的数值全部初始化为-1等。
通过tf.fill(shape， value)可以创建全为自定义数值Value的张量，形状有shape参数指定。
例如，创建元素为-1的标量：
In [30]:tf.fill([], -1) # 创建-1 的标量
Out[30]:
<tf.Tensor: id=124, shape=(), dtype=int32, numpy=-1>
例如，创建所有元素为-1 的向量：
In [31]:tf.fill([1], -1) # 创建-1 的向量
Out[31]:
<tf.Tensor: id=128, shape=(1,), dtype=int32, numpy=array([-1])>
例如，创建所有元素为 99 的矩阵：
In [32]:tf.fill([2,2], 99) # 创建 2 行 2 列，元素全为 99 的矩阵
Out[32]:
<tf.Tensor: id=136, shape=(2, 2), dtype=int32, numpy=
array([[99, 99],
      [99, 99]])>


4.4.4创建已知分布的张量
正态分布和均匀分布是最常见的分布，创建采样自这两种分布的张量非常有用，比如在卷积神经网络中，卷积核张量𝑾初始化为正态分布
有利于网络的训练；在对抗生成网络中，隐藏变量z一般采样自均匀分布。
通过tf.random.normal(shape, mean=0.0, stddev=1.0)可以创建形状为shape，均值为mean，标准差为stddev的正态分布𝒩(mean, stddev^2)。
例如，创建均值为 0，标准差为 1的正态分布：
In [33]: tf.random.normal([2,2]) # 创建标准正态分布的张量
Out[33]:
<tf.Tensor: id=143, shape=(2, 2), dtype=float32, numpy=
array([[-0.4307344 , 0.44147003],
       [-0.6563149 , -0.30100572]], dtype=float32)>
例如，创建均值为 1，标准差为 2 的正态分布：
In [34]: tf.random.normal([2,2], mean=1,stddev=2) # 创建正态分布的张量
Out[34]:
<tf.Tensor: id=150, shape=(2, 2), dtype=float32, numpy=
array([[-2.2687864, -0.7248812],
       [ 1.2752185, 2.8625617]], dtype=float32)>

通过tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32)可以创建采样自[minval， maxval)区间的均匀分布的张量。
例如创建采样自区间[0， 1)，shape为[2， 2]的矩阵：
In [35]: tf.random.uniform([2,2]) # 创建采样自[0,1)均匀分布的矩阵
Out[35]:
<tf.Tensor: id=158, shape=(2, 2), dtype=float32, numpy=
array([[0.65483284, 0.63064325],
 [0.008816 , 0.81437767]], dtype=float32)>

例如，创建采样自区间[0,10)，shape 为[2,2]的矩阵：
In [36]: tf.random.uniform([2,2],maxval=10) # 创建采样自[0,10)均匀分布的矩阵
Out[36]:
<tf.Tensor: id=166, shape=(2, 2), dtype=float32, numpy=
array([[4.541913 , 0.26521802],
       [2.578913 , 5.126876 ]], dtype=float32)>

如果需要均匀采样整形类型的数据，必须指定采样区间的最大值 maxval 参数，同时指定数据类型为 tf.int*型：
In [37]: # 创建采样自[0,100)均匀分布的整型矩阵
tf.random.uniform([2,2],maxval=100,dtype=tf.int32)
Out[37]:
<tf.Tensor: id=171, shape=(2, 2), dtype=int32, numpy=
array([[61, 21],
       [95, 75]])>


4.4.5创建序列
在循环计算或对张量进行索引时，经常需要创建一段连续的整形序列，可以通过tf.range()函数实现。tf.range(limit, delta=1)
可以创建[0,limit)之间，步长为delta的整形序列，不包含limit本身。
例如，创建0~10，步长为1的整型序列：
In [38]: tf.range(10) # 0~10，不包含 10
Out[38]:
<tf.Tensor: id=180, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>
例如，创建0~10，步长为2的整型序列：
In [39]: tf.range(10,delta=2)
Out[39]:
<tf.Tensor: id=185, shape=(5,), dtype=int32, numpy=array([0, 2, 4, 6, 8])>

通过tf.range(start, limit, delta=1)可以创建[start，limit)，步长为delta的序列，不包含limit本身。
In [40]: tf.range(1,10,delta=2) # 1~10
Out[40]:
<tf.Tensor: id=190, shape=(5,), dtype=int32, numpy=array([1, 3, 5, 7, 9])>
"""