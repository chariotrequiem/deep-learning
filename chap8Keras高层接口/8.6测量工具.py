# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 10:00
"""
在网络的训练过程中，经常需要统计准确率、召回率等测量指标，除了可以通过手动计算的方式获取这些统计数据外，
Keras 提供了一些常用的测量工具，位于 keras.metrics 模块中，专门用于统计训练过程中常用的指标数据。

Keras 的测量工具的使用方法一般有 4 个主要步骤：新建测量器，写入数据，读取统计数据和清零测量器

8.6.1新建测量器
在 keras.metrics 模块中，提供了较多的常用测量器类，如统计平均值的 Mean 类，统计准确率的 Accuracy 类，
统计余弦相似度的 CosineSimilarity 类等。下面我们以统计误差值为例。在前向运算时，我们会得到每一个 Batch 的平均误差，
但是我们希望统计每个Step 的平均误差，因此选择使用 Mean 测量器。新建一个平均测量器，代码如下：
# 新建平均测量器，适合 Loss 数据
loss_meter = metrics.Mean()


8.6.2写入数据
通过测量器的 update_state 函数可以写入新的数据，测量器会根据自身逻辑记录并处理采样数据。例如，在每个 Step 结束时采集一次 loss 值，
代码如下：
# 记录采样的数据，通过 float()函数将张量转换为普通数值
loss_meter.update_state(float(loss))
上述采样代码放置在每个 Batch 运算结束后，测量器会自动根据采样的数据来统计平均值。


8.6.3读取统计信息
在采样多次数据后，可以选择在需要的地方调用测量器的 result()函数，来获取统计值。
例如，间隔性统计 loss 均值，代码如下：
# 打印统计期间的平均 loss
print(step, 'loss:', loss_meter.result())


8.6.4清除状态
由于测量器会统计所有历史记录的数据，因此在启动新一轮统计时，有必要清除历史
状态。通过 reset_states()即可实现清除状态功能。例如，在每次读取完平均误差后，清零统
计信息，以便下一轮统计的开始，代码如下：
if step % 100 == 0:
    # 打印统计的平均 loss
    print(step, 'loss:', loss_meter.result())
    loss_meter.reset_states() # 打印完后，清零测量器


8.6.5准确率统计实战
按照测量工具的使用方法，我们利用准确率测量器 Accuracy 类来统计训练过程中的准确率。
首先新建准确率测量器，代码如下：
acc_meter = metrics.Accuracy() # 创建准确率测量器

在每次前向计算完成后，记录训练准确率数据。需要注意的是，Accuracy 类的 update_state函数的参数为预测值和真实值，而不是当前 Batch 的准确率。
我们将当前 Batch 样本的标签和预测结果写入测量器，代码如下：
# [b, 784] => [b, 10]，网络输出值
out = network(x)
# [b, 10] => [b]，经过 argmax 后计算预测值
pred = tf.argmax(out, axis=1)
pred = tf.cast(pred, dtype=tf.int32)
# 根据预测值与真实值写入测量器
acc_meter.update_state(y, pred)

在统计完测试集所有Batch的预测值后，打印统计的平均准确率，并清零测量器，代码如下：
# 读取统计结果
print(step, 'Evaluate Acc:', acc_meter.result().numpy())
acc_meter.reset_states() # 清零测量器


"""