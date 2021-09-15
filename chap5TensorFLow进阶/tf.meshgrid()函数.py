# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 21:30
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = tf.linspace(-8., 8, 100)  # 设置 x 轴的采样点
y = tf.linspace(-8., 8, 100)  # 设置 y 轴的采样点
x, y = tf.meshgrid(x, y)  # 生成网格点，并内部拆分后返回
# 打印拆分后的所有点的 x,y 坐标张量 shape
print(x.shape)
print(y.shape)
z = tf.sqrt(x**2 + y**2)
z = tf.sin(z)/z  # sinc函数实现
fig = plt.figure()
ax = Axes3D(fig)  # 设置3D坐标轴
ax.contour(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()