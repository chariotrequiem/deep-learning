# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/21 16:05
"""
在训练网络时，一般的流程是通过前向计算获得网络的输出值，再通过损失函数计算网络误差，然后通过自动求导工具计算梯度并更新，
同时间隔性地测试网络的性能。对于这种常用的训练逻辑，可以直接通过 Keras 提供的模型装配与训练等高层接口实现，简洁清晰。


8.2.1模型装配
在 Keras 中，有 2 个比较特殊的类：keras.Model 和 keras.layers.Layer 类。其中 Layer类是网络层的母类，
定义了网络层的一些常见功能，如添加权值、管理权值列表等。
Model 类是网络的母类，除了具有 Layer 类的功能，还添加了保存模型、加载模型、训练与测试模型等便捷功能。
Sequential 也是 Model 的子类，因此具有 Model 类的所有功能。
接下来介绍 Model 及其子类的模型装配与训练功能。我们以 Sequential 容器封装的网络为例，首先创建 5 层的全连接网络，
用于 MNIST 手写数字图片识别，代码如下：

# 创建 5 层的全连接网络
network = Sequential([layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)])
network.build(input_shape=(4, 28*28))
network.summary()

创建网络后，正常的流程是循环迭代数据集多个 Epoch，每次按批产生训练数据、前向计算，然后通过损失函数计算误差值，
并反向传播自动计算梯度、更新网络参数。这一部分逻辑由于非常通用，在 Keras 中提供了 compile()和 fit()函数方便实现上述逻辑。
首先通过compile 函数指定网络使用的优化器对象、损失函数类型，评价指标等设定，这一步称为装配。
例如：
# 导入优化器，损失函数模块
from tensorflow.keras import optimizers,losses
# 模型装配
# 采用 Adam 优化器，学习率为 0.01;采用交叉熵损失函数，包含 Softmax
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'] # 设置测量指标为准确率)
在 compile()函数中指定的优化器、损失函数等参数也是我们自行训练时需要设置的参数，并没有什么特别之处，
只不过 Keras 将这部分常用逻辑内部实现了，提高开发效率。


8.2.2模型训练
模型装配完成后，即可通过 fit()函数送入待训练的数据集和验证用的数据集，这一步称为模型训练。
例如：
# 指定训练集为 train_db，验证集为 val_db,训练 5 个 epochs，每 2 个 epoch 验证一次
# 返回训练轨迹信息保存在 history 对象中
history = network.fit(train_db, epochs=5, validation_data=val_db,
validation_freq=2)
其中 train_db 为 tf.data.Dataset 对象，也可以传入 Numpy Array 类型的数据；epochs 参数指定训练迭代的 Epoch 数量；
validation_data 参数指定用于验证(测试)的数据集和验证的频率validation_freq。

运行上述代码即可实现网络的训练与验证的功能，fit 函数会返回训练过程的数据记录history，其中 history.history 为字典对象，
包含了训练过程中的 loss、测量指标等记录项，我们可以直接查看这些训练数据，
例如：
In [4]: history.history # 打印训练记录
Out[4]: # 历史训练准确率
{'accuracy': [0.00011666667, 0.0, 0.0, 0.010666667, 0.02495],
'loss': [2465719710540.5845, # 历史训练误差
 78167808898516.03,
 404488834518159.6,
 1049151145155144.4,
 1969370184858451.0],
'val_accuracy': [0.0, 0.0], # 历史验证准确率
# 历史验证误差
'val_loss': [197178788071657.3, 1506234836955706.2]}
fit()函数的运行代表了网络的训练过程，因此会消耗相当的训练时间，并在训练结束后才返回，训练中产生的历史数据可以通过返回值对象取得。
可以看到通过 compile&fit 方 式实现的代码非常简洁和高效，大大缩减了开发时间。但是因为接口非常高层，灵活性也降低了，
是否使用需要用户自行判断。


8.2.3模型测试
Model 基类除了可以便捷地完成网络的装配与训练、验证，还可以非常方便的预测和测试。关于验证和测试的区别，我们会在过拟合一章详细阐述，
此处可以将验证和测试理 解为模型评估的一种方式。通过 Model.predict(x)方法即可完成模型的预测，
例如：

# 加载一个 batch 的测试数据
x,y = next(iter(db_test))
print('predict x:', x.shape) # 打印当前 batch 的形状
out = network.predict(x) # 模型预测，预测结果保存在 out 中
print(out)

其中 out 即为网络的输出。通过上述代码即可使用训练好的模型去预测新样本的标签信息。

如果只是简单的测试模型的性能，可以通过 Model.evaluate(db)循环测试完 db 数据集上所有样本，并打印出性能指标，
例如：
network.evaluate(db_test) # 模型测试，测试在 db_test 上的性能表现
"""