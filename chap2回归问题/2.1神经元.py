# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/13 15:20
"""
成年人大脑中包含了约 1000 亿个神经元，每个神经元通过树突获取输入信号，通过轴突传递输出信号，
神经元之间相互连接构成了巨大的神经网络，从而形成了人脑的感知和意识基础。
1943 年，心理学家沃伦·麦卡洛克(Warren McCulloch)和数理逻辑学家沃尔特·皮茨(Walter Pitts)通过对生物神经元的研究，
提出了模拟生物神经元机制的人工神经网络的数学模型，这一成果被美国神经学家弗兰克·罗森布拉特(Frank Rosenblatt)进一步发展
成感知机(Perceptron)模型，这也是现代深度学习的基石。

我们将从生物神经元的结构出发，重温科学先驱们的探索之路，逐步揭开自动学习机器的神秘面纱。
首先，我们把生物神经元(Neuron)的模型抽象为如图所示的数学结构：神经元输入向量𝒙 = [𝑥1,𝑥2,𝑥3,…,𝑥𝑛]T，
经过函数映射：𝑓𝜃: 𝒙 → 𝑦后得到输出𝑦，其中𝜃为函数𝑓自身的参数。考虑一种简化的情况，即线性变换：𝑓(𝒙) = 𝒘T𝒙 + 𝑏，展开为标量形式：
𝑓(𝒙) = 𝑤1𝑥1 + 𝑤2𝑥2 + 𝑤3𝑥3 + ⋯ + 𝑤𝑛𝑥𝑛 + 𝑏
参数𝜃 = {𝑤1, 𝑤2, 𝑤3, . . . , 𝑤𝑛, 𝑏}确定了神经元的状态，通过固定𝜃参数即可确定此神经元的处理逻辑。
当神经元输入节点数𝑛 = 1(单输入)时，神经元数学模型可进一步简化为：
𝑦 = 𝑤𝑥 + 𝑏
此时我们可以绘制出神经元的输出𝑦和输入𝑥的变化趋势，随着输入信号𝑥的增加，输出电平𝑦也随之线性增加,
其中𝑤参数可以理解为直线的斜率(Slope)，b参数为直线的偏置(Bias)。
对于某个神经元来说，𝑥和𝑦的映射关系𝑓𝑤,𝑏是未知但确定的。两点即可确定一条直线，为了估计𝑤和𝑏的值，
我们只需从图中直线上采样任意 2 个数据点:
可以看到，只需要观测两个不同数据点，就可完美求解单输入线性神经元模型的参数，
对于𝑁输入的线性神经元模型，只需要采样𝑁 + 1组不同数据点即可，似乎线性神经元模型可以得到完美解决。
那么上述方法存在什么问题呢？考虑对于任何采样点，都有可能存在观测误差，我们假设观测误差变量𝜖属于均值为𝜇，方差为𝜎2的正态分布
(Normal Distribution，或高斯分布，Gaussian Distribution)：𝒩(𝜇, 𝜎2)，则采样到的样本符合：
                         𝑦 = 𝑤𝑥 + 𝑏 + 𝜖, 𝜖~𝒩(𝜇, 𝜎2)
一旦引入观测误差后，即使简单如线性模型，如果仅采样两个数据点，可能会带来较大估计偏差。
为了减少观测误差引入的估计偏差，可以通过采样多组数据样本集合𝔻 = {(𝑥(1),𝑦(1)), (𝑥(2),𝑦(2)),… , (𝑥(𝑛), 𝑦(𝑛))}，
然后找出一条“最好”的直线，使得它尽可能地让所有采样点到该直线的误差(Error，或损失 Loss)之和最小。

也就是说，由于观测误差𝜖的存在，当我们采集了多个数据点𝔻时，可能不存在一条直线完美的穿过所有采样点。
退而求其次，我们希望能找到一条比较“好”的位于采样点中间的直线。那么怎么衡量“好”与“不好”呢？一个很自然的想法就是，
求出当前模型的所有采样点上的预测值𝑤𝑥(𝑖) + 𝑏与真实值𝑦(𝑖)之间的差的平方和作为总误差ℒ：
         ℒ = 1/𝑛 * ∑(𝑛,𝑖=1)(𝑤𝑥(𝑖) + 𝑏 − 𝑦(𝑖))2
然后搜索一组参数𝑤∗, 𝑏∗使得ℒ最小，对应的直线就是我们要寻找的最优直线.
其中𝑛表示采样点的个数。这种误差计算方法称为均方误差(Mean Squared Error，简称MSE）。
"""