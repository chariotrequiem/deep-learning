# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/14 9:48
"""
对于数值类型的张量，可以保存为不同字节长度的精度，如浮点数3.14既可以保存为16位(Bit)长度，也可以保存为32位甚至64位的精度。
位越长，精度越高，同时占用的内存空间也就越大。常用的精度类型有：
tf.int16, tf.int32, tf.int64, tf.float16, tf.float31, tf.float64等，其中tf.float64即为tf.double
在创建张量时，可以指定张量的保存精度，例如：
In [12]: # 创建指定精度的张量
tf.constant(123456789, dtype=tf.int16)
tf.constant(123456789, dtype=tf.int32)
Out[12]:
<tf.Tensor: id=33, shape=(), dtype=int16, numpy=-13035>
<tf.Tensor: id=35, shape=(), dtype=int32, numpy=123456789>
可以看到，保存精度过低时，数据123456789发生了溢出，得到了错误的结果，一般使用tf.int32、tf.int64精度。
对于浮点数，高精度的张量可以表示更准确的数据，例如采用tf.float32精度保存Π时，实际保存的数据为3.1415927。代码如下：
In [13]:
import numpy as np
np.pi # 从 numpy 中导入 pi 常量
tf.constant(np.pi, dtype=tf.float32) # 32 位
Out[13]:
<tf.Tensor: id=29, shape=(), dtype=float32, numpy=3.1415927>

如果采用tf.float64精度保存Π，则能获得更高的精度，实现如下：
In [14]: tf.constant(np.pi, dtype=tf.float64) # 64 位
Out[14]:
<tf.Tensor: id=31, shape=(), dtype=float64, numpy=3.141592653589793>

对于大部分深度学习算法，一般采用tf.int32和tf.float32可满足大部份场合的运算精度要求，部分对精度要求较高的算法，如强化学习某些算法，
可以选择使用tf.int64和tf.float64精度保存张量

4.2.1读取精度
通过访问张量的dtype成员属性可以判断张量的保存精度，例如：
In [15]:
print('before:',a.dtype) # 读取原有张量的数值精度
if a.dtype != tf.float32: # 如果精度不符合要求，则进行转换
 a = tf.cast(a,tf.float32) # tf.cast 函数可以完成精度转换
print('after :',a.dtype) # 打印转换后的精度
Out[15]:
before: <dtype: 'float16'>
after : <dtype: 'float32'>
对于某些只能处理指定精度类型的运算操作，需要提前检验输入张量的精度类型，并将不符合要求的张量进行类型转换。

4.2.2类型转换
系统的每个模块使用的数据类型、数值精度可能各不相同，对于不符合要求的张量的类型和精度，需要通过tf.cast函数进行转换，例如;
In [16]:
a = tf.constant(np.pi, dtype=tf.float16) # 创建 tf.float16 低精度张量
tf.cast(a, tf.double) # 转换为高精度张量
Out[16]:
<tf.Tensor: id=44, shape=(), dtype=float64, numpy=3.140625>

进行类型转换时，需要保证转换操作的合法性，例如将高精度的张量转换为低精度的张量时，可能发生数据溢出隐患：
In [17]:
a = tf.constant(123456789, dtype=tf.int32)
tf.cast(a, tf.int16) # 转换为低精度整型
Out[17]:
<tf.Tensor: id=38, shape=(), dtype=int16, numpy=-13035>

布尔类型和整形之间的相互转换也是合法的，是比较常见的操作：
In [18]:
a = tf.constant([True, False])
tf.cast(a, tf.int32) # 布尔类型转整型
Out[18]:
<tf.Tensor: id=48, shape=(2,), dtype=int32, numpy=array([1, 0])>
一般默认0表示False，1表示True，在TensorFlow中，将非0数字都视为True，例如：
In [19]:
a = tf.constant([-1, 0, 1, 2])
tf.cast(a, tf.bool) # 整型转布尔类型
Out[19]:
<tf.Tensor: id=51, shape=(4,), dtype=bool, numpy=array([ True, False, True, True])>

"""