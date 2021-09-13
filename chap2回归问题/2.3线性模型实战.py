# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/13 15:43
import numpy as np

"""
在介绍了用于优化𝑤和𝑏的梯度下降算法后，我们来实战训练单输入神经元线性模型。
首先我们需要采样自真实模型的多组数据，对于已知真实模型的玩具样例(Toy Example)，我们直接从指定的𝑤 = 1.477 , 𝑏 =0.089
的真实模型中直接采样：
            𝑦 = 1.477𝑥 + 0.089

1.采样数据
为了很好的模拟真实样本的观测误差，我们给模型添加误差自变量𝜖，他采样自均值为0，标准差为0.01的高斯分布：
         𝑦 = 1.477 𝑥 + 0.089+ 𝜖, 𝜖 ∼ 𝒩(0，0.01^2)
通过随机采样100次，我们获得n个样本的训练数据集D^train，代码如下：
"""
data = []  # 保存样本集的列表
for i in range(100):  # 循环采样100个点
    x = np.random.uniform(-10., 10.)  # 随机采样输入x
    # 采样高斯噪声
    eps = np.random.normal(0., 0.01)
    # 得到模型的输出
    y = 1.477 * x + 0.089 + eps
    data.append([x, y])  # 保存样本点
data = np.array(data)  # 转换为 2D Numpy数组
"""
循环进行 100 次采样，每次从均匀分布𝑈(−10 ,10)中随机采样一个数据𝑥，同时从均值为0，方差为0.1^2的高斯分布
𝒩(0，0.1^2)中随机采样噪声𝜖，根据真实模型生成𝑦的数据，并保存为 Numpy 数组

2. 计算误差
循环计算在每个点(𝑥(𝑖), 𝑦(𝑖))处的预测值与真实值之间差的平方并累加，从而获得训练集上的均方误差损失值。代码如下：
"""


def mse(b, w, points):
    # 根据当前的w和b参数计算均方差损失
    total_error = 0
    for i in range(0, len(points)):  # 循环迭代所有点
        x = points[i, 0]  # 获得i号点的输入x
        y = points[i, 1]  # 获得i号点的输出y
        # 计算差的方差并累加
        total_error += (y - (w * x + b)) ** 2
    # 将累加的误差求平均，得到均方差
    return total_error / float(len(points))


"""
最后的误差和除以数据样本总数，从而得到每个样本上的平均误差。

3. 计算梯度
根据之前介绍的梯度下降算法，我们需要计算出函数在每一个点上的梯度信息：(∂ℒ/∂𝑤 , ∂ℒ/∂𝑏)。
我们来推导一下梯度的表达式，首先考虑∂ℒ/∂𝑤，将均方差函数展开求偏导得：
∂ℒ/∂𝑤 = 2/𝑛 ∑(𝑤𝑥(𝑖) + 𝑏 − 𝑦(𝑖)) ∙ 𝑥(𝑖)
同理可得：
∂ℒ/∂𝑏 = 2/𝑛 ∑(𝑤𝑥(𝑖) + 𝑏 − 𝑦(𝑖)) 
根据偏导数的表达式，我们只需要计算在每一个点上面的(𝑤𝑥(𝑖) + 𝑏 − 𝑦(𝑖)) ∙ 𝑥(𝑖)和(𝑤𝑥(𝑖) + 𝑏 − 𝑦(𝑖))值，
平均后即可得到偏导数∂ℒ/∂𝑤和∂ℒ/∂𝑏。实现如下：
"""


def step_gradient(b_current, w_current, points, lr):
    # 计算误差函数在所有点上的导数，并更新w， b
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))  # 总样本数
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对b的导数：grad_b = 2(wx + b - y)
        b_gradient += (2/M) * ((w_current * x + b_current) - y)
        # 误差函数对w的导数：grad_w = 2(wx + b - y)*x
        w_gradient += (2/M) * x * ((w_current * x + b_current) - y)
    # 根据梯度下降算法更新w，b，其中lr为学习率
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


"""
4. 梯度更新
在计算出误差函数在𝑤和𝑏处的梯度后，我们可以根据式(2.1)来更新𝑤和𝑏的值。我们把对数据集的所有样本训练一次称为一个 Epoch，
共循环迭代 num_iterations 个 Epoch。实现如下：
"""


def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    # 循环更新w， b多次
    b = starting_b  # b的初始值
    w = starting_w  # w的初始值
    # 根据梯度下降算法更新多次
    for step in range(num_iterations):
        # 计算梯度并更新一次
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)  # 计算当前的均方差，用于监控训练进度
        if step % 50 == 0:  # 打印误差和实时的w，b值
            print(f"iteration:{step},loss:{loss}, w:{w}, b:{b}")
    return [b, w]  # 返回最后一次的w，b


# 主函数训练如下：
def main():
    # 加载训练集数据， 这些数据是通过真实模型添加误差采样得到的
    lr = 0.01  # 学习率
    initial_b = 0  # 初始化b为0
    initial_w = 0  # 初始化w为0
    num_iterations = 1000
    # 训练1000次，返回最优w*，b*和训练Loss的下降过程
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)  # 训练最优数值解w, b上的均方差
    print(f'Final loss:{loss}, w:{w}, b:{b}')


# 运行主程序
main()
"""
上述例子比较好地展示了梯度下降算法在求解模型参数上的强大之处。需要注意的是，对于复杂的非线性模型，
通过梯度下降算法求解到的𝑤和𝑏可能是局部极小值而非全局最小值解，这是由模型函数的非凸性决定的。但是我们在实践中发现，
通过梯度下降算法求得的数值解，它的性能往往都能优化得很好，可以直接使用求解到的数值解𝑤和𝑏来近似作为最优解。
"""