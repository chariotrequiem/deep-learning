# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 17:11
"""
本节我们将利用前面介绍的多层全连接网络的梯度推导结果，直接利用 Python 循环计算每一层的梯度，并按着梯度下降算法手动更新。
由于 TensorFlow 具有自动求导功能，我们选择没有自动求导功能的 Numpy 实现网络，并利用 Numpy 手动计算梯度并手动更新网络参数。

需要注意的是，本章推导的梯度传播公式是针对于多层全连接层，只有 Sigmoid 一种激活函数，并且损失函数为均方误差函数的网络类型。
对于其它类型的网络，比如激活函数采用 ReLU，损失函数采用交叉熵的网络，需要重新推导梯度传播表达式，但是方法是一样。
正是因为手动推导梯度的方法局限性较大，在实践中采用极少，更多的是利用自动求导工具计算。

我们将实现一个 4 层的全连接网络，来完成二分类任务。网络输入节点数为 2，隐藏层的节点数设计为：25、50和25，输出层两个节点，
分别表示属于类别 1 的概率和类别 2的概率，如图 7.13 所示。这里并没有采用 Softmax 函数将网络输出概率值之和进行约束，
而是直接利用均方误差函数计算与 One-hot 编码的真实标签之间的误差，所有的网络激活函数全部采用 Sigmoid 函数，
这些设计都是为了能直接利用我们的梯度传播公式。

7.9.1数据集
这里通过 scikit-learn 库提供的便捷工具生成 2000 个线性不可分的 2 分类数据集，数据的特征长度为 2，采样出的数据分布如图 7.14 所示，
所有的红色点为一类，所有的蓝色点为一类，可以看到每个类别数据的分布呈月牙状，并且是是线性不可分的，无法用线性网络获得较好效果。
为了测试网络的性能，我们按着7: 3比例切分训练集和测试集，其中2000 ∙ 0.3 = 600个样本点用于测试，不参与训练，剩下的 1400 个点用于网络的训练。

数据集的采集直接使用 scikit-learn 提供的 make_moons 函数生成，设置采样点数和切割比率，代码如下：
N_SAMPLES = 2000 # 采样点数
TEST_SIZE = 0.3 # 测试数量比率
# 利用工具函数直接生成数据集
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# 将 2000 个点按着 7:3 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)
可以通过如下可视化代码绘制数据集的分布，如图 7.14 所示。
# 绘制数据集的分布，X 为 2D 坐标，y 为数据点的标签
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None,dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1,cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral,edgecolors='none')

    plt.savefig('dataset.svg')
    plt.close()
# 调用 make_plot 函数绘制数据的分布，其中 X 为 2D 坐标，y 为标签
make_plot(X, y, "Classification Dataset Visualization ")
plt.show()


7.9.2网络层
通过新建类 Layer 实现一个网络层，需要传入网络层的输入节点数、输出节点数、激活函数类型等参数，
权值 weights 和偏置张量 bias 在初始化时根据输入、输出节点数自动生成并初始化。代码如下：

class Layer:
    # 全连接网络层
    def __init__(self, n_input, n_neurons, activation=None, weights=None,
                 bias=None):
        :param int n_input: 输入节点数
        :param int n_neurons: 输出节点数
        :param str activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        # 通过正态分布初始化网络权值，初始化非常重要，不合适的初始化将导致网络不收敛
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation  # 激活函数类型，如’sigmoid’
        self.last_activation = None  # 激活函数的输出值o
        self.error = None  # 用于计算当前层的delta 变量的中间变量
        self.delta = None  # 记录当前层的delta 变量，用于计算梯度

    # 网络层的前向传播函数实现如下，其中last_activation 变量用于保存当前层的输出值：
    def activate(self, x):
        # 前向传播函数
        r = np.dot(x, self.weights) + self.bias  # X@W+b
        # 通过激活函数，得到全连接层的输出o
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    # 上述代码中的self._apply_activation 函数实现了不同类型的激活函数的前向计算过程，
    # 尽管此处我们只使用Sigmoid 激活函数一种。代码如下：
    def _apply_activation(self, r):
        # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数，直接返回
        # ReLU 激活函数
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        # tanh 激活函数
        elif self.activation == 'tanh':
            return np.tanh(r)
        # sigmoid 激活函数
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    # 针对于不同类型的激活函数，它们的导数计算实现如下：
    def apply_activation_derivative(self, r):
        # 计算激活函数的导数
        # 无激活函数，导数为1
        if self.activation is None:
            return np.ones_like(r)
        # ReLU 函数的导数实现
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        # tanh 函数的导数实现
        elif self.activation == 'tanh':
            return 1 - r ** 2
        # Sigmoid 函数的导数实现
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r
可以看到，Sigmoid 函数的导数实现为𝑟 (1 − 𝑟)，其中𝑟即为𝜎(𝑧)



7.9.3网络模型
创建单层网络类后，我们实现网络模型的 NeuralNetwork 类，它内部维护各层的网络层 Layer 类对象，可以通过 add_layer 函数追加网络层，
实现创建不同结构的网络模型目的。代码如下：
class NeuralNetwork:
    # 神经网络模型大类
    def __init__(self):
        self._layers = [] # 网络层对象列表
    def add_layer(self, layer):
        # 追加网络层
        self._layers.append(layer)
    网络的前向传播只需要循环调各个网络层对象的前向计算函数即可，代码如下：
    def feed_forward(self, X):
        # 前向传播
        for layer in self._layers:
            # 依次通过各个网络层
            X = layer.activate(X)
        return X
根据图 7.13 的网络结构配置，利用 NeuralNetwork 类创建网络对象，并添加 4 层全连接层，代码如下：
nn = NeuralNetwork() # 实例化网络类
nn.add_layer(Layer(2, 25, 'sigmoid')) # 隐藏层 1, 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid')) # 隐藏层 2, 25=>50
nn.add_layer(Layer(50, 25, 'sigmoid')) # 隐藏层 3, 50=>25
nn.add_layer(Layer(25, 2, 'sigmoid')) # 输出层, 25=>2


网络模型的反向传播实现稍复杂，需要从最末层开始，计算每层的𝛿变量，然后根据推导出的梯度公式，
将计算出的𝛿变量存储在 Layer 类的 delta 变量中。代码如下：
def backpropagation(self, X, y, learning_rate):
    # 反向传播算法实现
    # 前向计算，得到输出值
    output = self.feed_forward(X)
    for i in reversed(range(len(self._layers))): # 反向循环
        layer = self._layers[i] # 得到当前层对象
        # 如果是输出层
        if layer == self._layers[-1]: # 对于输出层
            layer.error = y - output # 计算 2 分类任务的均方差的导数
            # 关键步骤：计算最后一层的 delta，参考输出层的梯度公式
            layer.delta = layer.error * layer.apply_activation_derivative(output)
        else: # 如果是隐藏层
            next_layer = self._layers[i + 1] # 得到下一层对象
            layer.error = np.dot(next_layer.weights, next_layer.delta)
            # 关键步骤：计算隐藏层的 delta，参考隐藏层的梯度公式
            layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
 …# 代码接下面
在反向计算完每层的𝛿变量后，只需要按着 𝜕ℒ 𝜕𝑤𝑖
= 𝑜𝑖𝛿 ( )公式计算每层参数的梯度，并
更新网络参数即可。由于代码中的 delta 计算的其实是−𝛿，因此更新时使用了加号。代码如下：
 def backpropagation(self, X, y, learning_rate):
 … # 代码接上面
 # 循环更新权值
 for i in range(len(self._layers)):
 layer = self._layers[i]
 # o_i 为上一网络层的输出
 o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
 # 梯度下降算法，delta 是公式中的负数，故这里用加号
 layer.weights += layer.delta * o_i.T * learning_rate
因此，在 backpropagation 函数中，反向计算每层的𝛿变量，并根据梯度公式计算每层参数的梯度值，按着梯度下降算法完成一次参数的更新。


7.9.4网络训练
这里的二分类任务网络设计为两个输出节点，因此需要将真实标签𝑦进行 One-hot 编码，代码如下：
def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
    # 网络训练函数
    # one-hot 编码
    y_onehot = np.zeros((y_train.shape[0], 2))
    y_onehot[np.arange(y_train.shape[0]), y_train] = 1
将 One-hot 编码后的真实标签与网络的输出计算均方误差，并调用反向传播函数更新网络参数，循环迭代训练集 1000 遍即可。代码如下：
    mses = []
    for i in range(max_epochs): # 训练 1000 个 epoch
        for j in range(len(X_train)): # 一次训练一个样本
            self.backpropagation(X_train[j], y_onehot[j], learning_rate)
        if i % 10 == 0:
            # 打印出 MSE Loss
            mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
            mses.append(mse)
            print('Epoch: #%s, MSE: %f' % (i, float(mse)))

            # 统计并打印准确率
            print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test), y_test.flatten()) * 100))
    return mses
"""