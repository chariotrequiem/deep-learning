# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/23 9:27
"""
我们已经介绍完卷积神经网络层的工作原理与实现方法，复杂的卷积神经网络模型也是基于卷积层的堆叠构成的。在过去的一段时间内，
研究人员发现网络层数越深，模型的表达能力越强，也就越有可能取得更好的性能。那么层层堆叠的卷积网络到底学到了什么特征，
使得层数越深，网络的表达能力越强呢？

2014 年，Matthew D. Zeiler 等人 [5]尝试利用可视化的方法去理解卷积神经网络到底学到了什么。
通过将每层的特征图利用“反卷积”网络(Deconvolutional Network)映射回输入图片，即可查看学到的特征分布，如图 10.32 所示。
可以观察到，第二层的特征对应到边、角、色彩等底层图像提取；第三层开始捕获到纹理这些中层特征；
第四、五层呈现了物体的部分特征，如小狗的脸部、鸟类的脚部等高层特征。通过这些可视化的手段，我们可以一定程度上感受卷积神经网络的特征学习过程。

图片数据的识别过程一般认为也是表示学习(Representation Learning)的过程，从接受到的原始像素特征开始，逐渐提取边缘、角点等底层特征，
再到纹理等中层特征，再到头部、物体部件等高层特征，最后的网络层基于这些学习到的抽象特征表示(Representation)做分类逻辑的学习。
学习到的特征越高层、越准确，就越有利于分类器的分类，从而获得较好的性能。从表示学习的角度来理解，卷积神经网络通过层层堆叠来逐层提取特征，
网络训练的过程可以看成特征的学习过程，基于学习到的高层抽象特征可以方便地进行分类任务。

应用表示学习的思想，训练好的卷积神经网络往往能够学习到较好的特征，这种特征的提取方法一般是通用的。
比如在猫、狗任务上学习到头、脚、身躯等特征的表示，在其它动物上也能够一定程度上使用。基于这种思想，
可以将在任务 A 上训练好的深层神经网络的前面数个特征提取层迁移到任务 B 上，只需要训练任务 B 的分类逻辑(表现为网络的最末数层)，
即可取得非常好的效果，这种方式是迁移学习的一种，从神经网络角度也称为网络微调(Fine-tuning)


"""