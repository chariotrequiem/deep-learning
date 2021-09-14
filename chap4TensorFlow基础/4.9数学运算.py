# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/14 21:50
"""
前面的章节已经使用了基本的加、减、乘、除等基本数学运算函数，本节我们将系统地介绍 TensorFlow 中常见的数学运算函数。

4.9.1加、减、乘、除运算
加、减、乘、除是最基本的数学运算，分别通过 tf.add, tf.subtract, tf.multiply, tf.divide函数实现，
TensorFlow 已经重载了+、 − 、 ∗ 、/运算符，一般推荐直接使用运算符来完成加、减、乘、除运算。

整除和余除也是常见的运算之一，分别通过//和%运算符实现。
我们来演示整除运算，例如：
import tensorflow as tf
a = tf.range(5)
b = tf.constant(2)
print(a)
print(b)
print(a//b)  # 整除运算
<tf.Tensor: id=115, shape=(5,), dtype=int32, numpy=array([0, 0, 1, 1, 2])>

余除运算，例如：
In [90]: a%b # 余除运算
Out[90]:
<tf.Tensor: id=117, shape=(5,), dtype=int32, numpy=array([0, 1, 0, 1, 0])>


4.9.2乘方运算
通过 tf.pow(x, a)可以方便地完成𝑦 = 𝑎的乘方运算，也可以通过运算符**实现 ∗∗ 𝑎运 算，实现如下：
In [91]:
x = tf.range(4)
tf.pow(x,3) # 乘方运算
Out[91]:
<tf.Tensor: id=124, shape=(4,), dtype=int32, numpy=array([ 0, 1, 8, 27])>
In [92]: x**2 # 乘方运算符
Out[92]:
<tf.Tensor: id=127, shape=(4,), dtype=int32, numpy=array([0, 1, 4, 9])>

设置指数为1/𝑎形式，即可实现𝑎√x根号运算，例如:
In [93]: x=tf.constant([1.,4.,9.])
x**(0.5) # 平方根
Out[93]:
<tf.Tensor: id=139, shape=(3,), dtype=float32, numpy=array([1., 2., 3.],
dtype=float32)>

特别地，对于常见的平方和平方根运算，可以使用 tf.square(x)和 tf.sqrt(x)实现。平方运算实现如下：
In [94]:x = tf.range(5)
x = tf.cast(x, dtype=tf.float32) # 转换为浮点数
x = tf.square(x) # 平方
Out[94]:
<tf.Tensor: id=159, shape=(5,), dtype=float32, numpy=array([ 0., 1., 4., 9., 16.], dtype=float32)>

平方根运算实现如下：
In [95]:tf.sqrt(x) # 平方根
Out[95]:
<tf.Tensor: id=161, shape=(5,), dtype=float32, numpy=array([0., 1., 2., 3., 4.], dtype=float32)>



4.9.3指数和对数运算
通过 tf.pow(a, x)或者**运算符也可以方便地实现指数运算𝑎𝑥，例如：
In [96]: x = tf.constant([1.,2.,3.])
2**x # 指数运算
Out[96]:
<tf.Tensor: id=179, shape=(3,), dtype=float32, numpy=array([2., 4., 8.], dtype=float32)>

特别地，对于自然指数e𝑥，可以通过 tf.exp(x)实现，例如：
In [97]: tf.exp(1.) # 自然指数运算
Out[97]:
<tf.Tensor: id=182, shape=(), dtype=float32, numpy=2.7182817>

在 TensorFlow 中，自然对数logex可以通过 tf.math.log(x)实现，例如:
In [98]: x=tf.exp(3.)
tf.math.log(x) # 对数运算
Out[98]:
<tf.Tensor: id=186, shape=(), dtype=float32, numpy=3.0>

如果希望计算其它底数的对数，可以根据对数的换底公式：
log𝑎 X=loge X/loge 𝑎
间接地通过 tf.math.log(x)实现。如计算log10 X 可以通过 loge X/loge 10
实现如下：
In [99]: x = tf.constant([1.,2.])
x = 10**x
tf.math.log(x)/tf.math.log(10.) # 换底公式
Out[99]:
<tf.Tensor: id=6, shape=(2,), dtype=float32, numpy=array([1., 2.],
dtype=float32)>
实现起来相对繁琐，也许 TensorFlow 以后会推出任意底数的 log 函数。


4.9.4矩阵相乘运算
神经网络中间包含了大量的矩阵相乘运算，前面我们已经介绍了通过@运算符可以方便的实现矩阵相乘，
还可以通过 tf.matmul(a, b)函数实现。需要注意的是，TensorFlow 中的矩阵相乘可以使用批量方式，
也就是张量𝑨和𝑩的维度数可以大于 2。当张量𝑨和𝑩维度数大于 2 时，TensorFlow 会选择𝑨和𝑩的最后两个维度进行矩阵相乘，
前面所有的维度都视作Batch 维度。

根据矩阵相乘的定义，𝑨和𝑩能够矩阵相乘的条件是，𝑨的倒数第一个维度长度(列)和𝑩的倒数第二个维度长度(行)必须相等。
比如张量 a shape:[4,3,28,32]可以与张量 b shape:[4,3,32,2]进行矩阵相乘，代码如下：
In [100]:
a = tf.random.normal([4,3,28,32])
b = tf.random.normal([4,3,32,2])
a@b # 批量形式的矩阵相乘
Out[100]:
<tf.Tensor: id=236, shape=(4, 3, 28, 2), dtype=float32, numpy=array([[[[-1.66706240e+00, -8.32602978e+00],
                                                                       [ 9.83304405e+00, 8.15909767e+00],
                                                                       [ 6.31014729e+00, 9.26124632e-01],…
得到shape为[4,3,28,2]的结果

矩阵相乘函数同样支持自动Broadcasting机制，例如：
In [101]:
a = tf.random.normal([4,28,32])
b = tf.random.normal([32,16])
tf.matmul(a,b) # 先自动扩展，再矩阵相乘
Out[101]:
<tf.Tensor: id=264, shape=(4, 28, 16), dtype=float32, numpy=
array([[[-1.11323869e+00, -9.48194981e+00, 6.48123884e+00, ...,6.53280640e+00, -3.10894990e+00, 1.53050375e+00],
         [ 4.35898495e+00, -1.03704405e+01, 8.90656471e+00, ...,
上述运算自动将变量b扩展为公共shape：[4, 32, 6], 再与变量a进行批量形式地矩阵相乘，得到的结果的shape为[4, 28, 16]
"""