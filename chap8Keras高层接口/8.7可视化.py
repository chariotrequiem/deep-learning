# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 10:18
"""
在网络训练的过程中，通过 Web 端远程监控网络的训练进度，可视化网络的训练结果，对于提高开发效率和实现远程监控是非常重要的。
TensorFlow 提供了一个专门的可视化工具，叫做 TensorBoard，它通过 TensorFlow 将监控数据写入到文件系统，并利用 Web后端监控对应的文件目录，
从而可以允许用户从远程查看网络的监控数据。

TensorBoard 的使用需要模型代码和浏览器相互配合。在使用 TensorBoard 之前，需要安装 TensorBoard 库，安装命令如下：
# 安装 TensorBoard
pip install tensorboard
接下来我们分模型端和浏览器端介绍如何使用 TensorBoard 工具监控网络训练进度。

8.7.1 模型端
在模型端，需要创建写入监控数据的 Summary 类，并在需要的时候写入监控数据。首先通过 tf.summary.create_file_writer 创建监控对象类实例，
并指定监控数据的写入目录，代码如下：
# 创建监控类，监控数据将写入 log_dir 目录
summary_writer = tf.summary.create_file_writer(log_dir)

我们以监控误差数据和可视化图片数据为例，介绍如何写入监控数据。在前向计算完成后，对于误差这种标量数据，
我们通过 tf.summary.scalar 函数记录监控数据，并指定时间戳 step 参数。
这里的 step 参数类似于每个数据对应的时间刻度信息，也可以理解为数据曲线的𝑥坐标，因此不宜重复。每类数据通过字符串名字来区分，
同类的数据需要写入相同名字的数据库中。例如：
with summary_writer.as_default(): # 写入环境
    # 当前时间戳 step 上的数据为 loss，写入到名为 train-loss 数据库中
    tf.summary.scalar('train-loss', float(loss), step=step)
TensorBoard 通过字符串 ID 来区分不同类别的监控数据，因此对于误差数据，我们将它命名为”train-loss”，其它类别的数据不可写入，防止造成数据污染。

对于图片类型的数据，可以通过 tf.summary.image 函数写入监控图片数据。例如，在训练时，可以通过 tf.summary.image 函数可视化样本图片。
由于 TensorFlow 中的张量一般包含了多个样本，因此 tf.summary.image 函数接受多个图片的张量数据，并通过设置max_outputs 参数来选择最多显示的图片数量，
代码如下：
with summary_writer.as_default():# 写入环境
    # 写入测试准确率
    tf.summary.scalar('test-acc', float(total_correct/total), step=step)
    # 可视化测试用的图片，设置最多可视化 9 张图片
    tf.summary.image("val-onebyone-images:", val_images, max_outputs=9, step=step)
运行模型程序，相应的数据将实时写入到指定文件目录中。


8.7.2浏览器端
在运行程序时，监控数据被写入到指定文件目录中。如果要实时远程查看、可视化这些数据，还需要借助于浏览器和 Web 后端。
首先是打开 Web 后端，通过在 cmd 终端运行tensorboard --logdir path 指定 Web 后端监控的文件目录 path，即可打开 Web 后端监控进程，
如图 8.2 所示：

此时打开浏览器，并输入网址 http://localhost:6006 (也可以通过 IP 地址远程访问，具体端口号可能会变动，可查看命令提示) 即可监控网络训练进度。
TensorBoard 可以同时显示多条监控记录，在监控页面的左侧可以选择监控记录，如图 8.3 所示：

在监控页面的上端可以选择不同类型数据的监控页面，比如标量监控页面SCALARS、图片可视化页面 IMAGES 等。
对于这个例子，我们需要监控的训练误差和测试准确率为标量类型数据，它的曲线在 SCALARS 页面可以查看，如图 8.4、图 8.5 所示。

在 IMAGES 页面，可以查看每个 Step 的图片可视化效果，如图 8.6 所示。
除了监控标量数据和图片数据外，TensorBoard 还支持通过 tf.summary.histogram 查看
张量数据的直方图分布，以及通过 tf.summary.text 打印文本信息等功能。例如：
with summary_writer.as_default():
    # 当前时间戳 step 上的数据为 loss，写入到 ID 位 train-loss 对象中
    tf.summary.scalar('train-loss', float(loss), step=step)
    # 可视化真实标签的直方图分布
    tf.summary.histogram('y-hist',y, step=step)
    # 查看文本信息
    tf.summary.text('loss-text',str(float(loss)))
在 HISTOGRAMS 页面即可查看张量的直方图，如图 8.7 所示，在 TEXT 页面可以查看文本信息，如图 8.8 所示。

实际上，除了 TensorBoard 工具可以无缝监控 TensorFlow 的模型数据外，Facebook 开发的 Visdom 工具同样可以方便可视化数据，
并且支持的可视化方式丰富，实时性高，使用起来较为方便。图 8.9 展示了 Visdom 数据的可视化方式。Visdom 可以直接接受PyTorch 的张量类型的数据，
但不能直接接受 TensorFlow 的张量类型数据，需要转换为Numpy 数组。对于追求丰富可视化手段和实时性监控的读者，Visdom 可能是更好的选择。
"""