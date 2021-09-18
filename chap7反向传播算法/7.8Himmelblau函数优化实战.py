# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 16:31
"""
Himmelblau 函数是用来测试优化算法的常用样例函数之一，它包含了两个自变量𝑥和 𝑦，数学表达式是：
                    𝑓(𝑥, 𝑦) = (𝑥2 + 𝑦 − 11)^2 + (𝑥 + 𝑦2 − 7)^2
首先我们通过如下代码实现 Himmelblau 函数的表达式：
def himmelblau(x):
    # himmelblau 函数实现，传入参数 x 为 2 个元素的 List
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
然后完成 Himmelblau 函数的可视化操作。通过 np.meshgrid 函数(TensorFlow 中也有meshgrid 函数)生成二维平面网格点坐标，代码如下：
x = np.arange(-6, 6, 0.1) # 可视化的 x 坐标范围为-6~6
y = np.arange(-6, 6, 0.1) # 可视化的 y 坐标范围为-6~6
print('x,y range:', x.shape, y.shape)
# 生成 x-y 平面采样网格点，方便可视化
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y]) # 计算网格点上的函数值

并利用 Matplotlib 库可视化 Himmelblau 函数，如图 7.11 所示，绘图代码如下：
# 绘制 himmelblau 函数曲面
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d') # 设置 3D 坐标轴
ax.plot_surface(X, Y, Z) # 3D 曲面图
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

大致可以看出，它共有 4 个局部极小值点，并且局部极小值都是 0，所以这 4 个局部极小值也是全局最小值。
我们可以通过解析的方法计算出局部极小值的精确坐标，它们分别是：
             (3,2), (−2 805, 3 131), (−3 779, −3 283), (3 584, −1 848)
在已经知晓极值解析解的情况下，我们现在利用梯度下降算法来优化 Himmelblau 函数的极小值数值解。

利用 TensorFlow 自动求导来求出函数在𝑥和𝑦的偏导数，并循环迭代更新𝑥和𝑦值，代码如下：
# 参数的初始化值对优化的影响不容忽视，可以通过尝试不同的初始化值，
# 检验函数优化的极小值情况
# [1., 0.], [-4, 0.], [4, 0.]
x = tf.constant([4., 0.]) # 初始化参数
for step in range(200):# 循环优化 200 次
    with tf.GradientTape() as tape: #梯度跟踪
        tape.watch([x]) # 加入梯度跟踪列表
        y = himmelblau(x) # 前向传播
    # 反向传播
    grads = tape.gradient(y, [x])[0]
    # 更新参数,0.01 为学习率
    x -= 0.01*grads
    # 打印优化的极小值
    if step % 20 == 19:
        print ('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))

经过 200 次迭代更新后，程序可以找到一个极小值解，此时函数值接近于 0。找到的数值解为：
step 199: x = [ 3.584428 -1.8481264], f(x) = 1.1368684856363775e-12
这与解析解之一(3 584, −1 848)几乎一样。
实际上，通过改变网络参数的初始化状态，程序可以得到多种极小值数值解。参数的初始化状态是可能影响梯度下降算法的搜索轨迹的，
甚至有可能搜索出完全不同的数值解

"""