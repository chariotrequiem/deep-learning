# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 16:29
"""
本节我们将利用全连接网络模型来完成汽车的效能指标 MPG(Mile Per Gallon，每加仑燃油英里数)的预测问题实战。

6.8.1数据集
我们采用 Auto MPG 数据集，它记录了各种汽车效能指标与气缸数、重量、马力等其它因子的真实数据，查看数据集的前 5 项，如表 6.1 所示，
其中每个字段的含义列在表 6.2 中。除了产地的数字字段表示类别外，其他字段都是数值类型。对于产地地段，1 表示美国，2 表示欧洲，3 表示日本。
                            表 6.1 Auto MPG 数据集前 5 项
MPG    Cylinders    Displacement     Horsepower    Weight   Acceleration   ModelYear   Origin
18.0      8             307.0          130.0       3504.0      12.0            70        1
15.0      8             350.0          165.0       3693.0      11.5            70        1
18.0      8             318.0          150.0       3436.0      11.0            70        1
16.0      8             304.0          150.0       3433.0      12.0            70        1
17.0      8             302.0          140.0       3449.0      10.5            70        1
                           表 6.2 数据集字段含义
MPG    Cylinders    Displacement    Horsepower     Weight    Acceleration   ModelYear   Origin
每加仑  气缸数           排量          马力         重量        加速度       型号年份    产地
燃油
英里

Auto MPG 数据集一共记录了 398 项数据，我们从 UCI 服务器下载并读取数据集到DataFrame 对象中，代码如下：
# 在线下载汽车效能数据集
dataset_path = keras.utils.get_file("auto-mpg.data",
"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# 利用 pandas 读取数据集，字段有效能（公里数每加仑），气缸数，排量，马力，重量
# 加速度，型号年份，产地
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,na_values = "?", comment='\t',sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
# 查看部分数据
dataset.head()
原始表格中的数据可能含有空字段(缺失值)的数据项，需要清除这些记录项：
dataset.isna().sum() # 统计空白数据
dataset = dataset.dropna() # 删除空白数据项
dataset.isna().sum() # 再次统计空白数据
清除后，观察到数据集记录项减为 392 项。
由于 Origin 字段为类别类型数据，我们将其移除，并转换为新的 3 个字段：USA、 Europe 和 Japan，分别代表是否来自此产地：
# 处理类别型数据，其中 origin 列代表了类别 1,2,3,分布代表产地：美国、欧洲、日本
# 先弹出(删除并返回)origin 这一列
origin = dataset.pop('Origin')
# 根据 origin 列来写入新的 3 个列
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail() # 查看新表格的后几项
按着 8:2 的比例切分数据集为训练集和测试集：
# 切分为训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
将 MPG 字段移出为标签数据：
# 移动 MPG 油耗效能这一列为真实标签 Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
统计训练集的各个字段数值的均值和标准差，并完成数据的标准化，通过 norm()函数
实现，代码如下：
# 查看训练集的输入 X 的统计数据
train_stats = train_dataset.describe()
train_stats.pop("MPG") # 仅保留输入 X
train_stats = train_stats.transpose() # 转置
# 标准化数据
def norm(x): # 减去每个字段的均值，并除以标准差
 return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset) # 标准化训练集
normed_test_data = norm(test_dataset) # 标准化测试集
打印出训练集和测试集的大小：
print(normed_train_data.shape,train_labels.shape)
print(normed_test_data.shape, test_labels.shape)
(314, 9) (314,) # 训练集共 314 行，输入特征长度为 9,标签用一个标量表示
(78, 9) (78,) # 测试集共 78 行，输入特征长度为 9,标签用一个标量表示利用切分的训练集数据构建数据集对象：
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,
train_labels.values)) # 构建 Dataset 对象
train_db = train_db.shuffle(100).batch(32) # 随机打散，批量化

我们可以通过简单地统计数据集中各字段之间的两两分布来观察各个字段对 MPG 的影响，如图 6.16 所示。
可以大致观察到，其中汽车排量、重量与 MPG 的关系比较简单，随着排量或重量的增大，汽车的 MPG 降低，能耗增加；
气缸数越小，汽车能做到的最好MPG 也越高，越可能更节能，这都是是符合我们的生活经验的。


6.8.2创建网络
考虑到 Auto MPG 数据集规模较小，我们只创建一个 3 层的全连接网络来完成 MPG值的预测任务。
输入𝑿的特征共有 9 种，因此第一层的输入节点数为 9。第一层、第二层的输出节点数设计为64和64，由于只有一种预测值，
输出层输出节点设计为 1。考虑MPG ∈ 𝑅+，因此输出层的激活函数可以不加，也可以添加 ReLU 激活函数。

我们将网络实现为一个自定义网络类，只需要在初始化函数中创建各个子网络层，并在前向计算函数 call 中实现自定义网络类的计算逻辑即可。
自定义网络类继承自keras.Model 基类，这也是自定义网络类的标准写法，以方便地利用 keras.Model 基类提供的 trainable_variables、save_weights
等各种便捷功能。网络模型类实现如下：
class Network(keras.Model):
    # 回归网络模型
    def __init__(self):
        super(Network, self).__init__()
        # 创建 3 个全连接层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # 依次通过 3 个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


6.8.3训练与测试
在完成主网络模型类的创建后，我们来实例化网络对象和创建优化器，代码如下：
model = Network() # 创建网络类实例
# 通过 build 函数完成内部张量的创建，其中 4 为任意设置的 batch 数量，9 为输入特征长度
model.build(input_shape=(4, 9))
model.summary() # 打印网络信息
optimizer = tf.keras.optimizers.RMSprop(0.001) # 创建优化器，指定学习率
接下来实现网络训练部分。通过 Epoch 和 Step 组成的双层循环训练网络，共训练 200个 Epoch，代码如下:
for epoch in range(200): # 200 个 Epoch
     for step, (x,y) in enumerate(train_db): # 遍历一次训练集
         # 梯度记录器，训练时需要使用它
         with tf.GradientTape() as tape:
         out = model(x) # 通过网络获得输出
         loss = tf.reduce_mean(losses.MSE(y, out)) # 计算 MSE
         mae_loss = tf.reduce_mean(losses.MAE(y, out)) # 计算 MAE
         if step % 10 == 0: # 间隔性地打印训练误差
         print(epoch, step, float(loss))
         # 计算梯度，并更新
         grads = tape.gradient(loss, model.trainable_variables)
         optimizer.apply_gradients(zip(grads, model.trainable_variables))


对于回归问题，除了 MSE 均方差可以用来模型的测试性能，还可以用平均绝对误差(Mean Absolute Error，简称 MAE)来衡量模型的性能，
它被定义为：
                mae ≜ 1 / 𝑑out ∑𝑖 |𝑦𝑖 − 𝑜𝑖|
程序运算时记录每个 Epoch 结束时的训练和测试 MAE 数据，并绘制变化曲线，如图 6.17所示。

可以观察到，在训练到约第 25 个 Epoch 时，MAE 的下降变得较缓慢，其中训练集的 MAE还在继续缓慢下降，但是测试集 MAE 几乎保持不变，
因此可以在约第 25 个 epoch 时提前结束训练，并利用此时的网络参数来预测新的输入样本即可。
"""