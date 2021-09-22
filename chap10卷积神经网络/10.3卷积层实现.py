# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 20:32
"""
在 TensorFlow 中，既可以通过自定义权值的底层实现方式搭建神经网络，也可以直接调用现成的卷积层类的高层方式快速搭建复杂网络。
我们主要以 2D 卷积为例，介绍如何实现卷积神经网络层。

10.3.1自定义权值
在 TensorFlow 中，通过 tf.nn.conv2d 函数可以方便地实现 2D 卷积运算。tf.nn.conv2d基于输入𝑿: [b, ℎ 𝑤 𝑐𝑖𝑛] 和卷积核𝑾: [𝑘 𝑘 𝑐𝑖𝑛 𝑐𝑜𝑢𝑡] 进行卷积运算，
得到输出𝑶 [b, ℎ′ 𝑤′ 𝑐𝑜𝑢𝑡] ，其中𝑐𝑖𝑛表示输入通道数，𝑐𝑜𝑢𝑡表示卷积核的数量，也是输出特征图的通道数。例如：
In [1]:
x = tf.random.normal([2,5,5,3]) # 模拟输入，3 通道，高宽为 5
# 需要根据[k,k,cin,cout]格式创建 W 张量，4 个 3x3 大小卷积核
w = tf.random.normal([3,3,3,4])
# 步长为 1, padding 为 0,
out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[0,0],[0,0],[0,0]])
Out[1]: # 输出张量的 shape
TensorShape([2, 3, 3, 4])

其中 padding 参数的设置格式为：
padding=[[0,0],[上,下],[左,右],[0,0]]
例如，上下左右各填充一个单位，则 padding 参数设置为 ，实现如下：
In [2]:
x = tf.random.normal([2,5,5,3]) # 模拟输入，3 通道，高宽为 5
# 需要根据[k,k,cin,cout]格式创建，4 个 3x3 大小卷积核
w = tf.random.normal([3,3,3,4])
# 步长为 1, padding 为 1,
out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[1,1],[1,1],[0,0]])
Out[2]: # 输出张量的 shape
TensorShape([2, 5, 5, 4])

特别地，通过设置参数 padding='SAME'、strides=1 可以直接得到输入、输出同大小的
卷积层，其中 padding 的具体数量由 TensorFlow 自动计算并完成填充操作。例如：
In [3]:
x = tf.random.normal([2,5,5,3]) # 模拟输入，3 通道，高宽为 5
w = tf.random.normal([3,3,3,4]) # 4 个 3x3 大小的卷积核
# 步长为,padding 设置为输出、输入同大小
# 需要注意的是, padding=same 只有在 strides=1 时才是同大小
out = tf.nn.conv2d(x,w,strides=1,padding='SAME')
Out[3]: TensorShape([2, 5, 5, 4])

当𝑠 > 时，设置 padding='SAME'将使得输出高、宽将成1/𝑠倍地减少。例如：
In [4]:
x = tf.random.normal([2,5,5,3])
w = tf.random.normal([3,3,3,4])
# 高宽先 padding 成可以整除 3 的最小整数 6，然后 6 按 3 倍减少，得到 2x2
out = tf.nn.conv2d(x,w,strides=3,padding='SAME')
Out [4]:TensorShape([2, 2, 2, 4])

卷积神经网络层与全连接层一样，可以设置网络带偏置向量。tf.nn.conv2d 函数是没有实现偏置向量计算的，添加偏置只需要手动累加偏置张量即可。
例如： # 根据[cout]格式创建偏置向量
b = tf.zeros([4])
# 在卷积输出上叠加偏置向量，它会自动 broadcasting 为[b,h',w',cout]
out = out + b



10.3.2卷积层类
通过卷积层类 layers.Conv2D 可以不需要手动定义卷积核𝑾和偏置𝒃张量，直接调用类实例即可完成卷积层的前向计算，实现更加高层和快捷。
在 TensorFlow 中，API 的命名有一定的规律，首字母大写的对象一般表示类，全部小写的一般表示函数，如 layers.Conv2D表示卷积层类，
nn.conv2d 表示卷积运算函数。使用类方式会(在创建类时或 build 时)自动创建需要的权值张量和偏置向量等，用户不需要记忆卷积核张量的定义格式，
因此使用起来更简单方便，但是灵活性也略低。函数方式的接口需要自行定义权值和偏置等，更加灵活和底层。

在新建卷积层类时，只需要指定卷积核数量参数 filters，卷积核大小 kernel_size，步长strides，填充 padding 等即可。
如下创建了 4 个3 × 3大小的卷积核的卷积层，步长为 1， padding 方案为'SAME'：
layer = layers.Conv2D(4,kernel_size=3,strides=1,padding='SAME')
如果卷积核高宽不等，步长行列方向不等，此时需要将 kernel_size 参数设计为 tuple格式(𝑘ℎ 𝑘𝑤)，strides 参数设计为(𝑠ℎ 𝑠𝑤)。
如下创建 4 个3 × 大小的卷积核，竖直方向移动步长𝑠ℎ = 2，水平方向移动步长𝑠𝑤 =1：
layer = layers.Conv2D(4,kernel_size=(3,4),strides=(2,1),padding='SAME')
创建完成后，通过调用实例(的__call__方法)即可完成前向计算，例如：
# 创建卷积层类
layer = layers.Conv2D(4,kernel_size=3,strides=1,padding='SAME')
out = layer(x) # 前向计算
out.shape # 输出张量的 shape
Out[5]:TensorShape([2, 5, 5, 4])

在类 Conv2D 中，保存了卷积核张量𝑾和偏置𝒃，可以通过类成员 trainable_variables直接返回𝑾和𝒃的列表。例如:
In [6]:
# 返回所有待优化张量列表
layer.trainable_variables
Out[6]:
[<tf.Variable 'conv2d/kernel:0' shape=(3, 3, 3, 4) dtype=float32, numpy=
array([[[[ 0.13485974, -0.22861657, 0.01000655, 0.11988598],
 [ 0.12811887, 0.20501086, -0.29820845, -0.19579397],
 [ 0.00858489, -0.24469738, -0.08591779, -0.27885547]], …
<tf.Variable 'conv2d/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0.,
0., 0.], dtype=float32)>]

通过调用 layer.trainable_variables 可以返回 Conv2D 类维护的𝑾和𝒃张量，这个类成员在获取网络层的待优化变量时非常有用。
也可以直接调用类实例 layer.kernel、layer.bias名访问𝑾和𝒃张量
"""