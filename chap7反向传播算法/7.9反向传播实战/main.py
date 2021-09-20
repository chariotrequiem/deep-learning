# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/20 15:29
import matplotlib.pyplot as plt

import Data
from layer import Layer
from neuralnetwork import NeuralNetwork

# 生成X，y以及训练集和测试集
X, y, X_train, X_test, y_train, y_test = Data.load_dataset()
# 将数据集可视化
Data.make_plot(X, y, "Classification Dataset Visualization ")

nn = NeuralNetwork()  # 实例化网络
nn.add_layer(Layer(2, 25, 'sigmoid'))  # 隐藏层 1, 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid'))  # 隐藏层 2, 25=>50
nn.add_layer(Layer(50, 25, 'sigmoid'))  # 隐藏层 3, 50=>25
nn.add_layer(Layer(25, 2, 'sigmoid'))  # 输出层, 25=>2
mses, accuracys = nn.train(X_train, X_test, y_train, y_test, 0.01, 1000)

x = [i for i in range(0, 101, 10)]

"""# 绘制MES曲线
plt.title('MES LOSS')
plt.plot(x, mses[:11], color='blue')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

# 绘制Accuracy曲线
plt.title("Accuracy")
plt.plot(x, accuracys[:11], color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()"""

# 绘制MES曲线和Accuracy曲线
plt.figure(figsize=(12, 6))  # 设置画布大小
plt.subplot(1, 2, 1)  # 1行2列，第一幅图片
plt.title('MES LOSS')
plt.plot(x, mses[:11], color='blue')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(x, accuracys[:11], color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()