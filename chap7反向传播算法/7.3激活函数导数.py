# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 10:47
"""7.3激活函数导数
这里介绍神经网络中常用的激活函数的导数推导。
7.3.1Sigmoid函数导数
回顾 Sigmoid 函数表达式：
                        𝜎(𝑥) = 1 / 1 +𝑒^(−𝑥)
我们来推导 Sigmoid 函数的导数表达式：
                        d𝜎(𝑥) / d𝑥  = d / d𝑥 (1 / 1 +𝑒^(−𝑥))
                                     = e−𝑥 / (1 + e−𝑥)2 = (1 + e−𝑥) − 1 / (1 + e−𝑥)2
                                    = 1 + 𝑒−𝑥 / (1 + 𝑒−𝑥)2 − (1 / 1 +𝑒^(−𝑥))^ 2 = 𝜎(𝑥) − 𝜎(𝑥)2 = 𝜎(1 − 𝜎)
可以看到，Sigmoid 函数的导数表达式最终可以表达为激活函数的输出值的简单运算，利用这一性质，在神经网络的梯度计算中，
通过缓存每层的 Sigmoid 函数输出值，即可在需要的时候计算出其导数。Sigmoid 函数的导数曲线如图 7.2 所示。

为了帮助理解反向传播算法的实现细节，本章选择不使用 TensorFlow 的自动求导功能，本章的实现部分全部使用 Numpy 演示，
将使用 Numpy 实现一个通过反向传播算法优化的多层神经网络。这里通过 Numpy 实现 Sigmoid 函数的导数，
代码如下:
import numpy as np # 导入 numpy 库
def sigmoid(x): # 实现 sigmoid 函数
    return 1 / (1 + np.exp(-x))
def derivative(x): # sigmoid 导数的计算
    # sigmoid 函数的表达式由手动推导而得
    return sigmoid(x)*(1-sigmoid(x))


7.3.2ReLU函数导数
回顾 ReLU 函数的表达式：
                    ReLU(𝑥) = max(0, 𝑥)
它的导数推导非常简单，直接可得：dd𝑥
                    ReLU = {1 𝑥 ≥ 0
                            0 𝑥 < 0
可以看到，ReLU 函数的导数计算简单，x 大于等于零的时候，导数值恒为 1，在反向传播过程中，它既不会放大梯度，
造成梯度爆炸(Gradient exploding)现象；也不会缩小梯度，造成梯度弥散(Gradient vanishing)现象。
ReLU 函数的导数曲线如图 7.3 所示。

在 ReLU 函数被广泛应用之前，神经网络中激活函数采用 Sigmoid 居多，但是 Sigmoid函数容易出现梯度弥散现象，当网络的层数增加后，
较前层的参数由于梯度值非常微小，参数长时间得不到有效更新，无法训练较深层的神经网络，导致神经网络的研究一直停留在浅层。
随着 ReLU 函数的提出，很好地缓解了梯度弥散的现象，神经网络的层数能够地达到较深层数，如 AlexNet 中采用了 ReLU 激活函数，
层数达到了 8 层，后续提出的上百层的卷积神经网络也多是采用 ReLU 激活函数。

通过 Numpy，我们可以方便地实现 ReLU 函数的导数，代码如下：
def derivative(x): # ReLU 函数的导数
    d = np.array(x, copy=True)  # 用于保存梯度的张量
    d[x < 0] = 0  # 元素为负的导数为 0
    d[x >= 0] = 1  # 元素为正的导数为 1
    return d


7.3.3LeakyReLU函数导数
回顾 LeakyReLU 函数的表达式：
                            LeakyReLU = { 𝑥 𝑥 ≥ 0
                                         𝑝𝑥 𝑥 < 0
它的导数可以推导为：
                            d / d𝑥 LeakyReLU = {1 𝑥 ≥ 0
                                                𝑝 𝑥 < 0
它和 ReLU 函数的不同之处在于，当 x 小于零时，LeakyReLU 函数的导数值并不为 0，而是常数𝑝，p 一般设置为某较小的数值，
如 0.01 或 0.02，LeakyReLU 函数的导数曲线如图 7.4 所示。
LeakyReLU 函数有效的克服了 ReLU 函数的缺陷，使用也比较广泛。我们可以通过Numpy 实现 LeakyReLU 函数的导数，代码如下：
# 其中 p 为 LeakyReLU 的负半段斜率，为超参数
def derivative(x, p):
    dx = np.ones_like(x) # 创建梯度张量，全部初始化为 1
    dx[x < 0] = p # 元素为负的导数为 p
    return dx


7.3.4Tanh函数梯度
回顾 tanh 函数的表达式：
                        tanh(𝑥) = (𝑒𝑥 − 𝑒−𝑥) / (𝑒𝑥 + 𝑒−𝑥)
                                = 2 ∙ sigmoid(2𝑥) − 1
它的导数推导为：
                        d / d𝑥 tanh(𝑥) = (𝑒𝑥 + 𝑒−𝑥)(𝑒𝑥 + 𝑒−𝑥) − (𝑒𝑥 − 𝑒−𝑥)(𝑒𝑥 − 𝑒−𝑥) / (𝑒𝑥 + 𝑒−𝑥)2
                                       = 1 − (𝑒𝑥 − 𝑒−𝑥)2 / (𝑒𝑥 + 𝑒−𝑥)2 = 1 − tanh2(𝑥)
tanh 函数及其导数曲线如图 7.5 所示。

在 Numpy 中，借助于 Sigmoid 函数实现 Tanh 函数的导数，代码如下：
def sigmoid(x): # sigmoid 函数实现
    return 1 / (1 + np.exp(-x))
def tanh(x): # tanh 函数实现
    return 2*sigmoid(2*x) - 1
def derivative(x): # tanh 导数实现
    return 1-tanh(x)**2"""