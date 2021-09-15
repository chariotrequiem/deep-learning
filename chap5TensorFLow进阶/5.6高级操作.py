# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 16:51
"""
上述介绍的操作函数大部分都是常有并且容易理解的，接下来我们将介绍部分常用，但是稍复杂的功能函数。

5.6.1 tf.gather
tf.gather 可以实现根据索引号收集数据的目的。考虑班级成绩册的例子，假设共有 4个班级，每个班级 35 个学生，8 门科目，
保存成绩册的张量 shape 为[4,35,8]。
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32) # 成绩册张量
现在需要收集第 1~2 个班级的成绩册，可以给定需要收集班级的索引号：[0,1]，并指定班级的维度 axis=0，
通过 tf.gather 函数收集数据，代码如下:
In [38]:tf.gather(x,[0,1],axis=0) # 在班级维度收集第 1~2 号班级成绩册
Out[38]:<tf.Tensor: id=83, shape=(2, 35, 8), dtype=int32, numpy=
                                                                array([[[43, 10, 93, 85, 75, 87, 28, 19],
                                                                         [52, 17, 44, 88, 82, 54, 16, 65],
                                                                         [98, 26, 1, 47, 59, 3, 59, 70],…

实际上，对于上述需求，通过切片𝑥[: 2]可以更加方便地实现。但是对于不规则的索引方式，比如，
需要抽查所有班级的第 1、4、9、12、13、27 号同学的成绩数据，则切片方式实现起来非常麻烦，
而 tf.gather 则是针对于此需求设计的，使用起来更加方便，实现如下：
In [39]: # 收集第 1,4,9,12,13,27 号同学成绩
tf.gather(x,[0,3,8,11,12,26],axis=1)
Out[39]:<tf.Tensor: id=87, shape=(4, 6, 8), dtype=int32, numpy=array([[[43, 10, 93, 85, 75, 87, 28, 19],
                                                                        [74, 11, 25, 64, 84, 89, 79, 85],...

如果需要收集所有同学的第 3 和第 5 门科目的成绩，则可以指定科目维度 axis=2，实现如下：
In [40]:tf.gather(x,[2,4],axis=2) # 第 3，5 科目的成绩
Out[40]:<tf.Tensor: id=91, shape=(4, 35, 2), dtype=int32, numpy=array([[[93, 75],
                                                                        [44, 82],
                                                                        [ 1, 59],…
可以看到，tf.gather非常适合索引号没有规则的场合，其中索引号可以乱序排列，此时收集的数据也是对应顺序，例如：
In [41]:a=tf.range(8)
a=tf.reshape(a,[4,2]) # 生成张量 a
Out[41]:<tf.Tensor: id=115, shape=(4, 2), dtype=int32, numpy=
array([[0, 1],
 [2, 3],
 [4, 5],
 [6, 7]])>
In [42]:tf.gather(a,[3,1,0,2],axis=0) # 收集第 4,2,1,3 号元素
Out[42]:<tf.Tensor: id=119, shape=(4, 2), dtype=int32, numpy=array([[6, 7],
                                                                    [2, 3],
                                                                    [0, 1],
                                                                    [4, 5]])>
我们将问题变得稍微复杂一点。如果希望抽查第[2,3]班级的第[3,4,6,27]号同学的科目成绩，
则可以通过组合多个 tf.gather 实现。首先抽出第[2,3]班级，实现如下：
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32) # 成绩册张量
In [43]:
students=tf.gather(x,[1,2],axis=0) # 收集第 2,3 号班级
Out[43]:<tf.Tensor: id=227, shape=(2, 35, 8), dtype=int32, numpy=array([[[ 0, 62, 99, 7, 66, 56, 95, 98],…
再从这 2 个班级的同学中提取对应学生成绩，代码如下：
In [44]: # 基于 students 张量继续收集
tf.gather(students,[2,3,5,26],axis=1) # 收集第 3,4,6,27 号同学
Out[44]:<tf.Tensor: id=231, shape=(2, 4, 8), dtype=int32, numpy=array([[[69, 67, 93, 2, 31, 5, 66, 65], …
此时得到这 2 个班级 4 个同学的成绩张量，shape 为[2, 4, 8]

我们继续问题进一步复杂化。这次我们希望抽查第 2 个班级的第 2 个同学的所有科目，第 3 个班级的第 3 个同学的所有科目，
第 4 个班级的第 4 个同学的所有科目。那么怎么实现呢？
可以通过笨方式，一个一个的手动提取数据。首先提取第一个采样点的数据：𝑥[1,1],可得到 8 门科目的数据向量:
In [45]: x[1,1] # 收集第 2 个班级的第 2 个同学
Out[45]:<tf.Tensor: id=236, shape=(8,), dtype=int32, numpy=array([45, 34,
99, 17, 3, 1, 43, 86])>
再串行提取第二个采样点的数据：𝑥[2,2]，以及第三个采样点的数据𝑥[3,3]，最后通过 stack方式合并采样结果，实现如下：
In [46]: tf.stack([x[1,1],x[2,2],x[3,3]],axis=0)
Out[46]:<tf.Tensor: id=250, shape=(3, 8), dtype=int32, numpy=
                                                            array([[45, 34, 99, 17, 3, 1, 43, 86],
                                                             [11, 25, 84, 95, 97, 95, 69, 69],
                                                             [ 0, 89, 52, 29, 76, 7, 2, 98]])>

这种方法也能正确地得到 shape 为[3,8]的结果，其中 3 表示采样点的个数，4 表示每个采样点的数据，看上去似乎也不错。
但是它最大的问题在于手动串行方式地执行采样，计算效率极低。有没有更好的方式实现呢？
这就是下一节要介绍的 tf.gather_nd 的功能。



5.6.2tf.gather_nd
通过 tf.gather_nd 函数，可以通过指定每次采样点的多维坐标来实现采样多个点的目的。回到上面的挑战，
我们希望抽查第 2 个班级的第 2 个同学的所有科目，第 3 个班级的第 3 个同学的所有科目，
第 4 个班级的第 4 个同学的所有科目。那么这 3 个采样点的索引坐标可以记为：[1,1]、[2,2]、[3,3]，
我们将这个采样方案合并为一个 List 参数，即[[1,1],[2,2],[3,3]]，通过 tf.gather_nd 函数即可，实现如下：
In [47]: # 根据多维坐标收集数据
tf.gather_nd(x,[[1,1],[2,2],[3,3]])
Out[47]:<tf.Tensor: id=256, shape=(3, 8), dtype=int32, numpy=
array([[45, 34, 99, 17, 3, 1, 43, 86],
 [11, 25, 84, 95, 97, 95, 69, 69],
 [ 0, 89, 52, 29, 76, 7, 2, 98]])>
可以看到，结果与串行采样的方式完全一致，实现更简洁，计算效率大大提升。

一般地，在使用 tf.gather_nd 采样多个样本时，例如希望采样𝑖号班级，𝑗个学生，𝑘门科目的成绩，
则可以表达为[. . . ,[𝑖,𝑗, 𝑘], . . .]，外层的括号长度为采样样本的个数，内层列表包含了每个采样点的索引坐标，例如：
In [48]: # 根据多维度坐标收集数据
tf.gather_nd(x,[[1,1,2],[2,2,3],[3,3,4]])
Out[48]:<tf.Tensor: id=259, shape=(3,), dtype=int32, numpy=array([99, 95, 76])>
上述代码中，我们抽出了班级 1 的学生 1 的科目 2、班级 2 的学生 2 的科目 3、班级 3 的学 生 3 的科目 4 的成绩，
共有 3 个成绩数据，结果汇总为一个 shape 为[3]的张量。


5.6.3tf.boolean_mask
除了可以通过给定索引号的方式采样，还可以通过给定掩码(Mask)的方式进行采样。继续以 shape 为[4,35,8]的成绩册张量为例，
这次我们以掩码方式进行数据提取。考虑在班级维度上进行采样，对这 4 个班级的采样方案的掩码为
mask = [True, False, False, True]
即采样第 1 和第 4 个班级的数据，通过 tf.boolean_mask(x, mask, axis)可以在 axis 轴上根据mask 方案进行采样，
实现为：
In [49]: # 根据掩码方式采样班级，给出掩码和维度索引
tf.boolean_mask(x,mask=[True, False,False,True],axis=0)
Out[49]:<tf.Tensor: id=288, shape=(2, 35, 8), dtype=int32, numpy=array([[[43, 10, 93, 85, 75, 87, 28, 19],…
注意掩码的长度必须与对应维度的长度一致，如在班级维度上采样，则必须对这 4 个班级是否采样的掩码全部指定，掩码长度为 4。

如果对 8 门科目进行掩码采样，设掩码采样方案为
mask = [True, False, False, True, True, False, False, True]
即采样第 1、4、5、8 门科目，则可以实现为：
In [50]: # 根据掩码方式采样科目
tf.boolean_mask(x,mask=[True,False,False,True,True,False,False,True],axis=2)
Out[50]:<tf.Tensor: id=318, shape=(4, 35, 4), dtype=int32, numpy=
array([[[43, 85, 75, 19],…
不难发现，这里的 tf.boolean_mask 的用法其实与 tf.gather 非常类似，只不过一个通过掩码方式采样，一个直接给出索引号采样。

现在我们来考虑与 tf.gather_nd 类似方式的多维掩码采样方式。为了方便演示，我们将班级数量减少到 2 个，
学生的数量减少到 3 个，即一个班级只有 3 个学生，shape 为 [2,3,8]。如果希望采样第 1 个班级的第 1~2 号学生，
第 2 个班级的第 2~3 号学生，通过tf.gather_nd 可以实现为：
In [51]:x = tf.random.uniform([2,3,8],maxval=100,dtype=tf.int32)
tf.gather_nd(x,[[0,0],[0,1],[1,1],[1,2]]) # 多维坐标采集
Out[51]:<tf.Tensor: id=325, shape=(4, 8), dtype=int32, numpy=
array([[52, 81, 78, 21, 50, 6, 68, 19],
 [53, 70, 62, 12, 7, 68, 36, 84],
 [62, 30, 52, 60, 10, 93, 33, 6],
 [97, 92, 59, 87, 86, 49, 47, 11]])>
共采样 4 个学生的成绩，shape 为[4, 8]。

如果用掩码方式，怎么表达呢？如下表 5.2 所示，行为每个班级，列为每个学生，表中数据表达了对应位置的采样情况：
                 表 5.2 成绩册掩码采样方案
                学生 0      学生 1     学生 2
        班级 0  True         True      False
        班级 1  False        True      True
因此，通过这张表，就能很好地表征利用掩码方式的采样方案，代码实现如下：
In [52]: # 多维掩码采样
tf.boolean_mask(x,[[True,True,False],[False,True,True]])
Out[52]:<tf.Tensor: id=354, shape=(4, 8), dtype=int32, numpy=
array([[52, 81, 78, 21, 50, 6, 68, 19],
 [53, 70, 62, 12, 7, 68, 36, 84],
 [62, 30, 52, 60, 10, 93, 33, 6],
 [97, 92, 59, 87, 86, 49, 47, 11]])>
采样结果与 tf.gather_nd 完全一致。可见 tf.boolean_mask 既可以实现了 tf.gather 方式的一维掩码采样，
又可以实现 tf.gather_nd 方式的多维掩码采样。
上面的 3 个操作比较常用，尤其是 tf.gather 和 tf.gather_nd 出现的频率较高，必须掌握。下面再补充 3 个高阶操作。



5.6.4tf.where
通过 tf.where(cond, a, b)操作可以根据 cond 条件的真假从参数𝑨或𝑩中读取数据，条件判定规则如下：
                     𝑜𝑖 = {𝑎𝑖 cond𝑖为 True
                           𝑏𝑖 cond𝑖为 False
其中𝑖为张量的元素索引，返回的张量大小与𝑨和𝑩一致，当对应位置的cond𝑖为 True，𝑜𝑖从 𝑎𝑖中复制数据；
当对应位置的cond𝑖为 False，𝑜𝑖从𝑏𝑖中复制数据。考虑从 2 个全 1 和全 0 的 3 × 3大小的张量𝑨和𝑩中提取数据，
其中cond𝑖为 True 的位置从𝑨中对应位置提取元素 1，cond𝑖为 False 的位置从𝑩对应位置提取元素 0，代码如下：
a = tf.ones([3,3]) # 构造 a 为全 1 矩阵
b = tf.zeros([3,3]) # 构造 b 为全 0 矩阵
# 构造采样条件
cond =
tf.constant([[True,False,False],[False,True,False],[True,True,False]])
tf.where(cond,a,b) # 根据条件从 a,b 中采样
Out[53]:<tf.Tensor: id=384, shape=(3, 3), dtype=float32, numpy=
array([[1., 0., 0.],
       [0., 1., 0.],
       [1., 1., 0.]], dtype=float32)>\
可以看到，返回的张量中为 1 的位置全部来自张量 a，返回的张量中为 0 的位置来自张量b。
当参数 a=b=None 时，即 a 和 b 参数不指定，tf.where 会返回 cond 张量中所有 True 的元素的索引坐标。考虑如下 cond 张量：
In [54]: cond # 构造的 cond 张量
Out[54]:<tf.Tensor: id=383, shape=(3, 3), dtype=bool, numpy=
array([[ True, False, False],
       [False, True, False],
       [ True, True, False]])>
其中 True 共出现 4 次，每个 True 元素位置处的索引分别为[0,0]、[1,1]、[2,0]、[2,1]，
可以直接通过 tf.where(cond)形式来获得这些元素的索引坐标，代码如下：
In [55]:tf.where(cond) # 获取 cond 中为 True 的元素索引
Out[55]:<tf.Tensor: id=387, shape=(4, 2), dtype=int64, numpy=
array([[0, 0],
 [1, 1],
 [2, 0],
 [2, 1]], dtype=int64)>


那么这有什么用途呢？考虑一个场景，我们需要提取张量中所有正数的数据和索引。首先构造张量 a，并通过比较运算得到所有正数的位置掩码：
In [56]:x = tf.random.normal([3,3]) # 构造 a
Out[56]:<tf.Tensor: id=403, shape=(3, 3), dtype=float32, numpy=
array([[-2.2946844 , 0.6708417 , -0.5222212 ],
       [-0.6919401 , -1.9418817 , 0.3559235 ],
       [-0.8005251 , 1.0603906 , -0.68819374]], dtype=float32)>
通过比较运算，得到所有正数的掩码：
In [57]:mask=x>0 # 比较操作，等同于 tf.math.greater()
mask
Out[57]:<tf.Tensor: id=405, shape=(3, 3), dtype=bool, numpy=array([[False, True, False],
                                                             [False, False, True],
                                                             [False, True, False]])>
通过 tf.where 提取此掩码处 True 元素的索引坐标：
In [58]:indices=tf.where(mask) # 提取所有大于 0 的元素索引
Out[58]:<tf.Tensor: id=407, shape=(3, 2), dtype=int64, numpy=array([[0, 1],
                                                                    [1, 2],
                                                                    [2, 1]], dtype=int64)>
拿到索引后，通过 tf.gather_nd 即可恢复出所有正数的元素：
In [59]:tf.gather_nd(x,indices) # 提取正数的元素值
Out[59]:<tf.Tensor: id=410, shape=(3,), dtype=float32, numpy=array([0.6708417, 0.3559235, 1.0603906], dtype=float32)>
实际上，当我们得到掩码 mask 之后，也可以直接通过 tf.boolean_mask 获取所有正数的元 素向量:
In [60]:tf.boolean_mask(x,mask) # 通过掩码提取正数的元素值
Out[60]:<tf.Tensor: id=439, shape=(3,), dtype=float32, numpy=array([0.6708417, 0.3559235, 1.0603906], dtype=float32)>
结果也是一致的。

通过上述一系列的比较、索引号收集和掩码收集的操作组合，我们能够比较直观地感受到这个功能是有很大的实际应用的，
并且深刻地理解它们的本质有利于更加灵活地选用简便高效的方式实现我们的目的。


5.6.5scatter_nd
通过 tf.scatter_nd(indices, updates, shape)函数可以高效地刷新张量的部分数据，但是这个函数只能在全 0 的白板张量上面执行刷新操作，
因此可能需要结合其它操作来实现现有张量的数据刷新功能。
如下图 5.3 所示，演示了一维张量白板的刷新运算原理。白板的形状通过 shape 参数表示，需要刷新的数据索引号通过 indices 表示，
新数据为 updates。根据 indices 给出的索引位置将 updates 中新的数据依次写入白板中，并返回更新后的结果张量。

我们实现一个图 5.3 中向量的刷新实例，代码如下：
In [61]: # 构造需要刷新数据的位置参数，即为 4、3、1 和 7 号位置
indices = tf.constant([[4], [3], [1], [7]])
# 构造需要写入的数据，4 号位写入 4.4,3 号位写入 3.3，以此类推
updates = tf.constant([4.4, 3.3, 1.1, 7.7])
# 在长度为 8 的全 0 向量上根据 indices 写入 updates 数据
tf.scatter_nd(indices, updates, [8])
Out[61]:<tf.Tensor: id=467, shape=(8,), dtype=float32, numpy=array([0. ,
1.1, 0. , 3.3, 4.4, 0. , 0. , 7.7], dtype=float32)>
可以看到，在长度为 8 的白板上，写入了对应位置的数据，4 个位置的数据被刷新。

考虑 3 维张量的刷新例子，如下图 5.4 所示，白板张量的 shape 为[4,4,4]，共有 4 个通道的特征图，每个通道大小为4 × 4，
现有 2 个通道的新数据 updates:[2,4,4]，需要写入索引为[1,3]的通道上。
我们将新的特征图写入现有白板张量，实现如下：
In [62]: # 构造写入位置，即 2 个位置
indices = tf.constant([[1],[3]])
updates = tf.constant([# 构造写入数据，即 2 个矩阵
 [[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],
 [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]])
# 在 shape 为[4,4,4]白板上根据 indices 写入 updates
tf.scatter_nd(indices,updates,[4,4,4])
Out[62]:<tf.Tensor: id=477, shape=(4, 4, 4), dtype=int32, numpy=
array([[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]],
 [[5, 5, 5, 5], # 写入的新数据 1
 [6, 6, 6, 6],
 [7, 7, 7, 7],
 [8, 8, 8, 8]],
 [[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]],
 [[1, 1, 1, 1], # 写入的新数据 2
 [2, 2, 2, 2],
 [3, 3, 3, 3],
 [4, 4, 4, 4]]])>
可以看到，数据被刷新到第 2 和第 4 个通道特征图上。



5.6.6meshgrid
通过 tf.meshgrid 函数可以方便地生成二维网格的采样点坐标，方便可视化等应用场合。考虑 2 个自变量 x 和 y 的 Sinc 函数表达式为：
                          𝑧 =sin(𝑥^2 + 𝑦^2) / 𝑥^2 + 𝑦^2
如果需要绘制在𝑥 ∈ [−8,8],𝑦 ∈ [−8,8]区间的 Sinc 函数的 3D 曲面，如图 5.5 所示，则首先需要生成 x 和 y 轴的网格点坐标集合{(𝑥, 𝑦)}，
这样才能通过 Sinc 函数的表达式计算函数在每个(𝑥, 𝑦)位置的输出值 z。可以通过如下方式生成 1 万个坐标采样点：
points = [] # 保存所有点的坐标列表
for x in range(-8,8,100): # 循环生成 x 坐标，100 个采样点
    for y in range(-8,8,100): # 循环生成 y 坐标，100 个采样点
        z = sinc(x,y) # 计算每个点(x,y)处的 sinc 函数值
        points.append([x,y,z]) # 保存采样点
很明显这种串行采样方式效率极低，那么有没有简洁高效地方式生成网格坐标呢？答案是肯定的，tf.meshgrid 函数即可实现。

通过在 x 轴上进行采样 100 个数据点，y 轴上采样 100 个数据点，然后利用
tf.meshgrid(x, y)即可返回这 10000 个数据点的张量数据，保存在 shape 为[100,100,2]的张量
中。为了方便计算，tf.meshgrid 会返回在 axis=2 维度切割后的 2 个张量𝑨和𝑩，其中张量𝑨
包含了所有点的 x 坐标，𝑩包含了所有点的 y 坐标，shape 都为[100,100]，实现如下：
In [63]:
x = tf.linspace(-8.,8,100) # 设置 x 轴的采样点
y = tf.linspace(-8.,8,100) # 设置 y 轴的采样点
x,y = tf.meshgrid(x,y) # 生成网格点，并内部拆分后返回
x.shape,y.shape # 打印拆分后的所有点的 x,y 坐标张量 shape
Out[63]: (TensorShape([100, 100]), TensorShape([100, 100]))

利用生成的网格点坐标张量𝑨和𝑩，Sinc 函数在 TensorFlow 中实现如下：
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z # sinc 函数实现
通过 matplotlib 库即可绘制出函数在𝑥 ∈ [−8,8],𝑦 ∈ [−8,8]区间的 3D 曲面，如图 5.5 中所示。代码如下：
import matplotlib
from matplotlib import pyplot as plt
# 导入 3D 坐标轴支持
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig) # 设置 3D 坐标轴
# 根据网格点绘制 sinc 函数 3D 曲面
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()
"""