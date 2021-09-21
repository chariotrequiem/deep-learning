# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 17:44
from tensorflow.keras import layers


class MyDense(layers.Layer):
    # 自定义网络层
    def __init__(self, inp_dim, outp_dim):
        super().__init__()
        # 创建权值张量并添加到类管理列表中，设置为需要优化
        self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)


net = MyDense(4, 3)  # 创建输入为4，输出为3结点的自定义层
# 查看自定义层的参数列表(类的全部参数列表， 类的待优化参数列表)
print(net.variables, net.trainable_variables)