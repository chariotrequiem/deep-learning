# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/14 10:39
"""
为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，TensorFlow 增加了一种专门的数据类型来支持梯度信息的记录：
tf.Variable。
tf.Variable 类型在普通的张量类型基础上添加了 name，trainable 等属性来支持计算图的构建。
由于梯度运算会消耗大量的计算资源，而且会自动更新相关参数，对于不需要的优化的张量，如神经网络的输入𝑿，不需要通过tf.Variable封装；
相反，对于需要计算梯度并优化的张量，如神经网络层的𝑾 和𝒃，需要通过 tf.Variable 包裹以便 TensorFlow 跟踪相关梯度信息。
通过 tf.Variable()函数可以将普通张量转换为待优化张量，例如：
In [20]:
a = tf.constant([-1, 0, 1, 2]) # 创建 TF 张量
aa = tf.Variable(a) # 转换为 Variable 类型
aa.name, aa.trainable # Variable 类型张量的属性
Out[20]:
('Variable:0', True)
其中张量的 name 和 trainable 属性是 Variable 特有的属性，name 属性用于命名计算图中的变量，这套命名体系是 TensorFlow 内部维护的，
一般不需要用户关注 name 属性；trainable属性表征当前张量是否需要被优化，创建 Variable 对象时是默认启用优化标志，可以设置
trainable=False 来设置张量不需要优化。

除了通过普通张量方式创建Variable，也可以直接创建，例如：
In [21]:
a = tf.Variable([[1,2],[3,4]]) # 直接创建 Variable 张量
Out[21]:
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=array([[1, 2], [3, 4]])>

待优化张量可视为普通张量的特殊类型，普通张量其实也可以通过 GradientTape.watch()方法临时加入跟踪梯度信息的列表，从而支持自动求导功能
"""