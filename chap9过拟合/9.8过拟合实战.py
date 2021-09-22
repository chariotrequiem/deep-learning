# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 14:45
"""
前面我们大量使用了月牙形状的 2 分类数据集来演示网络模型在各种防止过拟合措施下的性能表现。
本节实战我们将基于月牙形状的 2 分类数据集的过拟合与欠拟合模型，进行完整的实战。

9.8.1构建数据集
我们使用的数据集样本特性向量长度为2，标签为0或1，分别代表了两种类别。
借助于scikit-learn 库中提供的 make_moons 工具，我们可以生成任意多数据的训练集。
首先打开 cmd 命令终端，安装 scikit-learn 库，命令如下：
# pip 安装 scikit-learn 库
pip install -U scikit-learn

为了演示过拟合现象，我们只采样了 1000 个样本数据，同时添加标准差为 0.25 的高 斯噪声数据。代码如下：
# 导入数据集生成工具
from sklearn.datasets import make_moons
# 从 moon 分布中随机采样 1000 个点，并切分为训练集-测试集
X, y = make_moons(n_samples = N_SAMPLES, noise=0.25, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = TEST_SIZE, random_state=42)
"""