# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 16:56
import tensorflow as tf
from tensorflow.keras import losses
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# 在线下载汽车效能数据集
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# 读取数据集，字段有效能（公里数每加仑），气缸数，排量，马力，重量，加速度，型号年份，产地
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.head())  # 查看部分数据
# 原始表格中的数据可能含有空字段(缺失值)的数据项，需要清除这些记录项：
dataset.isna().sum()  # 统计空白数据
dataset = dataset.dropna()  # 删除空白数据项
dataset.isna().sum()  # 再次统计空白数据
print(dataset)  # 清楚后，观察到数据集记录项减为392项

# 处理类别型数据，其中origin列代表了类别1，2，3，分别代表产地：美国、欧洲、日本
# 先弹出(删除并返回)origin这一列
origin = dataset.pop('Origin')
# 根据 origin 列来写入新的 3 个列
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())  # 查看新表格的后几项

# 切分数据集为训练集与测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 移动 MPG 油耗效能这一列为真实标签 Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 统计训练集的各个字段数值的均值和标准差，并完成数据的标准化，通过norm()函数实现
# 查看训练集的输入x的统计数据
train_stats = train_dataset.describe()
print(train_stats)
# train_stats.pop("MPG")  # 仅保留输入X  # 书上有本句， 但MPG实际上已经在前面弹出
train_stats = train_stats.transpose()  # 转置
print(train_stats)


# 标准化数据
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)  # 标准化训练集
normed_test_data = norm(test_dataset)  # 标准化测试集

# 打印训练集和测试集的大小：
print(normed_train_data.shape, train_labels.shape)  # (314, 9)  (314, )
print(normed_test_data.shape, test_labels.shape)  # (78, 9)  (78, )

# 利用切分的训练集数据构建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))  # 构建Dataset对象
train_db = train_db.shuffle(1000).batch(32)  # 随机打散，批量化


class Network(keras.Model):
    # 回归网络模型
    def __init__(self):
        super(Network, self).__init__()
        # 创建3个全连接层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # 依次通过3个全连接层
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


model = Network()  # 创建网络类实例

# 通过build函数完成内部张量的创建，其中4为任意设置的batch数量，9为输入特征长度
model.build(input_shape=(32, 9))
print(model.summary())  # 打印网络信息
optimizer = tf.keras.optimizers.RMSprop(0.001)  # 创建优化器，指定学习率


mae = []
# 网络训练部分
# 通过Epoch和Step组成的双层循环训练网络，共训练200个Epoch
for epoch in range(200):  # 200个Epoch
    for step, (x, y) in enumerate(train_db):  # 遍历一次训练集
        # 梯度记录器，训练时需要使用
        with tf.GradientTape() as tape:
            out = model(x)  # 通过网络获得输出
            loss = tf.reduce_mean(losses.MSE(y, out))  # 计算MSE
            mae_loss = tf.reduce_mean(losses.MAE(y, out))  # 计算MAE

        if step % 10 == 0:  # 间隔性的打印训练误差
            print(epoch, step, float(loss))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


