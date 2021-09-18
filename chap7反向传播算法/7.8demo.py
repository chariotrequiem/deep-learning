# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 16:38
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(x):
    # himmelblau 函数实现，传入参数 x 为 2 个元素的 List
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = tf.linspace(-6., 6, 1000)  # 可视化的 x 坐标范围为-6~6
y = tf.linspace(-6., 6, 1000)  # 可视化的 y 坐标范围为-6~6
print('x, y range: ', x.shape, y.shape)
# 生成 x-y 平面采样网格点，方便可视化
X, Y = tf.meshgrid(x, y)
print('X, Y maps: ', X.shape, Y.shape)

z = himmelblau([X, Y])   # 计算网格点上的函数值
# 绘制 himmelblau 函数曲面
fig = plt.figure('himmelblau')
ax = Axes3D(fig)  # 设置3D坐标轴
ax.plot_surface(X, Y, z, cmap=plt.get_cmap('rainbow'), edgecolor='black')  # 3D曲面图
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.contourf(X,Y,z,zdir='Z',offset=-2,cmap='rainbow')  将z轴压到xoy平面上
plt.show()

# 函数的初始化值对优化的影响不容忽视，可以通过尝试不同的初始化值
# 检验函数优化的极小值情况
# [1., 0.], [-4, 0.], [4, 0.]
x = tf.constant([4., 0])  # 初始化参数

for step in range(200):  # 循环优化200次
    with tf.GradientTape() as tape:  # 梯度跟踪
        tape.watch([x])  # 加入梯度跟踪列表
        y = himmelblau(x)  # 前向传播
    # 反向传播
    grads = tape.gradient(y, [x])[0]
    # 更新参数，0.01为学习率
    x -= 0.01 * grads
    # 打印优化的极小值
    if step % 20 == 19:
        print('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))