# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 17:35
"""
尽管 Keras 提供了很多的常用网络层类，但深度学习可以使用的网络层远远不止这些。科研工作者一般是自行实现了较为新颖的网络层，
经过大量实验验证有效后，深度学习框架才会跟进，内置对这些网络层的支持。因此掌握自定义网络层、网络的实现非常重要。

对于需要创建自定义逻辑的网络层，可以通过自定义类来实现。在创建自定义网络层类时，需要继承自 layers.Layer 基类；创建自定义的网络类时，
需要继承自 keras.Model 基类，这样建立的自定义类才能够方便的利用 Layer/Model 基类提供的参数管理等功能，
同时也能够与其他的标准网络层类交互使用。

8.4.1自定义网络层
对于自定义的网络层，至少需要实现初始化__init__方法和前向传播逻辑 call 方法。我们以某个具体的自定义网络层为例，
假设需要一个没有偏置向量的全连接层，即 bias 为 0，同时固定激活函数为 ReLU 函数。尽管这可以通过标准的 Dense 层创建，
但我们还是通过实现这个“特别的”网络层类来阐述如何实现自定义网络层。

首先创建类，并继承自 Layer 基类。创建初始化方法，并调用母类的初始化函数，由于是全连接层，因此需要设置两个参数：
输入特征的长度 inp_dim 和输出特征的长度outp_dim，并通过 self.add_variable(name, shape)创建 shape 大小，
名字为 name 的张量𝑾， 并设置为需要优化。代码如下：
class MyDense(layers.Layer):
    # 自定义网络层
    def __init__(self, inp_dim, outp_dim):
        super().__init__()
        # 创建权值张量并添加到类管理列表中，设置为需要优化
        self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)

需要注意的是，self.add_variable会返回张量w的python引用，而变量名name有TensorFlow内部维护，使用的比较好。
我们实例化MyDense类，并查看其参数列表，例如：
net = MyDense(4, 3)  # 创建输入为4，输出为3结点的自定义层
# 查看自定义层的参数列表(类的全部参数列表， 类的待优化参数列表)
print(net.variables, net.trainable_variables)
可以看到𝑾张量被自动纳入类的参数列表。

通过修改为 self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=False)，我们可以设置𝑾张量不需要被优化，
此时再来观测张量的管理状态：
([<tf.Variable 'w:0' shape=(4, 3) dtype=float32, numpy=…], # 类的全部参数列表
[])# 类的不需要优化参数列表
可以看到，此时张量并不会被 trainable_variables 管理。
此外，类初始化中创建为 tf.Variable类型的类成员变量也会自动纳入张量管理中，例如：
# 通过 tf.Variable 创建的类成员也会自动加入类参数列表
self.kernel = tf.Variable(tf.random.normal([inp_dim, outp_dim]), trainable=False)
打印出管理的张量列表如下： # 类的参数列表
([<tf.Variable 'Variable:0' shape=(4, 3) dtype=float32, numpy=…],
[])# 类的不需要优化参数列表

完成自定义类的初始化工作后，我们来设计自定义类的前向运算逻辑，对于这个例子，只需要完成𝑶 = 𝑿@𝑾矩阵运算，
并通过固定的 ReLU 激活函数即可，代码如下:
def call(self, inputs, training=None):
    # 实现自定义类的前向计算逻辑
    # X@W
    out = inputs @ self.kernel
    # 执行激活函数运算
    out = tf.nn.relu(out)
    return out
如上所示，自定义类的前向运算逻辑实现在 call(inputs, training=None)函数中，其中 inputs代表输入，由用户在调用时传入；
training 参数用于指定模型的状态：training 为 True 时执行训练模式，training 为 False 时执行测试模式，默认参数为 None，即测试模式。

由于全连接层的训练模式和测试模式逻辑一致，此处不需要额外处理。对于部份测试模式和训练模式不一致的网络层，需要根据 training 参数来设计需要执行的逻辑。



8.4.2自定义网络
在完成了自定义的全连接层类实现之后，我们基于上述的“无偏置的全连接层”来实现 MNIST手写数字图片模型的创建。
自定义网络类可以和其他标准类一样，通过 Sequential 容器方便地封装成一个网络模型：
network = Sequential([MyDense(784, 256), # 使用自定义的层
                    MyDense(256, 128),
                    MyDense(128, 64),
                    MyDense(64, 32),
                    MyDense(32, 10)])
network.build(input_shape=(None, 28*28))
network.summary()
可以看到，通过堆叠我们的自定义网络层类，一样可以实现 5 层的全连接层网络，每层全连接层无偏置张量，同时激活函数固定地使用 ReLU 函数。

Sequential 容器适合于数据按序从第一层传播到第二层，再从第二层传播到第三层，以此规律传播的网络模型。对于复杂的网络结构，
例如第三层的输入不仅是第二层的输出，还有第一层的输出，此时使用自定义网络更加灵活。下面我们来创建自定义网络类，首先创建类，
并继承自 Model 基类，分别创建对应的网络层对象，代码如下：
class MyModel(keras.Model):
    # 自定义网络类，继承自 Model 基类
    def __init__(self):
        super(MyModel, self).__init__()
        # 完成网络内需要的网络层的创建工作
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)
然后实现自定义网络的前向运算逻辑，代码如下：
    def call(self, inputs, training=None):
        # 自定义前向运算逻辑
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
这个例子可以直接使用第一种方式，即 Sequential 容器包裹实现。但自定义网络的前向计 算逻辑可以自由定义，更为通用，我们会在卷积神经网络一章看到自定义网络的优越性。
"""