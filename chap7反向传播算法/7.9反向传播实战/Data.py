# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/20 14:42
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False


def load_dataset():
    # 采样点数
    N_SAMPLES = 2000
    # 测试数量比率
    TEST_SIZE = 0.3
    # 利用工具函数直接生成数据集
    # noise：默认是false，数据集是否加入高斯噪声
    # random_state：生成随机种子，给定一个int型数据，能够保证每次生成数据相同
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
    # 将2000个点按照7：3分割为训练集和测试集
    # 该函数按照用户设定的比例，随即将样本集合划分训练集和测试集，并返回划分好的训练集和测试集数据
    # test_size 若在0~1之间，为测试集样本数目与原始样本数目之比；若为整数，则是测试集样本的数目。
    # random_state 随机数种子
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return X, y, X_train, X_test, y_train, y_test


def make_plot(X, y, plot_name, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')  # 白色网格
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)  # 调整子图布局
    plt.subplots_adjust(right=0.80)  # left、bottom、right、top围成的区域就是子图的区域
    if XX is not None and YY is not None and preds is not None:
        # 画等高线（会填充轮廓）  # 画出等高线，核心函数是plt.contourf()
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=plt.cm.Spectral)
        # 画等高线（绘制轮廓线）
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')
    plt.show()


if __name__ == '__main__':
    X, y, X_train, X_test, y_train, y_test = load_dataset()
    # 调用 make_plot 函数绘制数据的分布，其中 X 为 2D 坐标， y 为标签
    make_plot(X, y, "Classification Dataset Visualization ")