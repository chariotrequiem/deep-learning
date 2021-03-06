# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/26 14:24
"""
现在我们来正式介绍生成对抗网络的网络结构和训练方法。

13.2.1网络结构
生成对抗网络包含了两个子网络：生成网络(Generator，简称 G)和判别网络(Discriminator，简称 D)，其中生成网络 G 负责学习样本的真实分布，
判别网络 D 负责将生成网络采样的样本与真实样本区分开来。

生成网络G(𝒛) 生成网络 G 和自编码器的 Decoder 功能类似，从先验分布𝑝𝒛(∙)中采样隐藏变量𝒛~𝑝𝒛(∙)，通过生成网络 G 参数化的𝑝𝑔(𝒙|𝒛)分布，
获得生成样本𝒙~𝑝𝑔(𝒙|𝒛)，如图13.3 所示。其中隐藏变量𝒛的先验分布𝑝𝒛(∙)可以假设为某中已知的分布，比如多元均匀分布𝑧~Uniform(−1,1)。

𝑝𝑔(𝒙|𝒛)可以用深度神经网络来参数化，如下图 13.4 所示，从均匀分布𝑝𝒛(∙)中采样出隐藏变量𝒛，经过多层转置卷积层网络参数化的𝑝𝑔(𝒙|𝒛)分布中采样出样本𝒙𝑓。
从输入输出层面来看，生成器 G 的功能是将隐向量𝒛通过神经网络转换为样本向量𝒙𝑓，下标𝑓代表假样本(Fake samples)。

判别网络D(𝒙) 判别网络和普通的二分类网络功能类似，它接受输入样本𝒙的数据集，包含了采样自真实数据分布𝑝𝑟(∙)的样本𝒙𝑟~𝑝𝑟(∙)，
也包含了采样自生成网络的假样本𝒙𝑓~𝑝𝑔(𝒙|𝒛)，𝒙𝑟和𝒙𝑓共同组成了判别网络的训练数据集。判别网络输出为𝒙属于真实样本的概率𝑃(𝒙为真|𝒙)，
我们把所有真实样本𝒙𝑟的标签标注为真(1)，所有生成网络产生的样本𝒙𝑓标注为假(0)，通过最小化判别网络 D 的预测值与标签之间的误差来优化判别网络参数，
如图 13.5 所示。

13.2.2网络训练
GAN 博弈学习的思想体现在在它的训练方式上，由于生成器 G 和判别器 D 的优化目标不一样，不能和之前的网络模型的训练一样，只采用一个损失函数。
下面我们来分别介绍如何训练生成器 G 和判别器 D。

对于判别网络 D，它的目标是能够很好地分辨出真样本𝒙𝑟与假样本𝒙𝑓。以图片生成为例，它的目标是最小化图片的预测值和真实值之间的交叉熵损失函数：
                                        𝑚𝑖𝑛 ℒ = CE(𝐷𝜃(𝒙𝑟),𝑦𝑟,𝐷𝜃(𝒙𝑓), 𝑦𝑓)
其中𝐷𝜃(𝒙𝑟)代表真实样本𝒙𝑟在判别网络𝐷𝜃的输出，𝜃为判别网络的参数集，𝐷𝜃(𝒙𝑓)为生成样本𝒙𝑓在判别网络的输出，𝑦𝑟为𝒙𝑟的标签，
由于真实样本标注为真，故𝑦𝑟 = 1，𝑦𝑓为生成样本的𝒙𝑓的标签，由于生成样本标注为假，故𝑦𝑓 = 0。CE 函数代表交叉熵损失函数CrossEntropy。
二分类问题的交叉熵损失函数定义为：
                                ℒ = − ∑ log𝐷𝜃(𝒙𝑟)  − ∑ log (1 − 𝐷𝜃(𝒙𝑓))
                                    𝒙𝑟~𝑝𝑟(∙)           𝒙𝑓~𝑝𝑔(∙)
因此判别网络 D 的优化目标是：
                                𝜃∗ = argmin − ∑ log𝐷𝜃(𝒙𝑟)  − ∑ log (1 − 𝐷𝜃(𝒙𝑓))
                                            𝒙𝑟~𝑝𝑟(∙)           𝒙𝑓~𝑝𝑔(∙)
把𝑚𝑖𝑛ℒ问题转换为𝑚𝑎𝑥 −ℒ，并写成期望形式：
                            𝜃∗ = argmax𝔼𝒙𝑟~𝑝𝑟(∙) log𝐷𝜃(𝒙𝑟) + 𝔼𝒙𝑓~𝑝𝑔(∙) log (1 − 𝐷𝜃(𝒙𝑓))

对于生成网络G(𝒛)，我们希望𝒙𝑓 = 𝐺(𝒛)能够很好地骗过判别网络 D，假样本𝒙𝑓在判别网络的输出越接近真实的标签越好。也就是说，
在训练生成网络时，希望判别网络的输出𝐷(𝐺(𝒛))越逼近 1 越好，最小化𝐷(𝐺(𝒛))与 1 之间的交叉熵损失函数：
                                min ℒ = 𝐶𝐸 (𝐷 (𝐺𝜙(𝒛)) , 1) = −log𝐷 (𝐺𝜙(𝒛))
                                 𝜙
把min ℒ问题转换成max−ℒ，并写成期望形式：
   𝜙               𝜙

                            𝜙∗ = argmax𝔼𝒛~𝑝𝑧(∙)log𝐷 (𝐺𝜙(𝒛))
                                    𝜙

再次等价转化为： 𝜙∗ = argminℒ = 𝔼𝒛~𝑝𝑧(∙)log[1 − 𝐷(𝐺𝜙(𝒛))]
                        𝜙

其中𝜙为生成网络 G 的参数集，可以利用梯度下降算法来优化参数𝜙。


13.2.3统一目标函数
我们把判别网络的目标和生成网络的目标合并，写成min − max博弈形式：
min max ℒ(𝐷,𝐺) = 𝔼𝒙𝑟~𝑝𝑟(∙)log𝐷𝜃(𝒙𝑟) + 𝔼𝒙𝑓~𝑝𝑔(∙)log (1 − 𝐷𝜃(𝒙𝑓))
𝜙   𝜃          = 𝔼𝒙~𝑝𝑟(∙)log𝐷𝜃(𝒙) + 𝔼𝒛~𝑝𝑧(∙)log (1 − 𝐷𝜃(𝐺𝜙(𝒛)))

算法流程如下:
算法 1：GAN 训练算法
随机初始化参数𝜽和𝝓
repeat
    for k 次 do
        随机采样隐向量𝒛~𝒑𝒛(∙)
        随机采样真实样本𝒙 ~𝒑 (∙)
        根据梯度上升算法更新 D 网络：
            𝛁𝜽𝔼𝒙 ~𝒑 (∙)𝐥𝐨𝐠𝑫𝜽(𝒙 ) + 𝔼𝒙𝒇~𝒑𝒈(∙)𝐥𝐨𝐠 (𝟏 − 𝑫𝜽(𝒙𝒇))
    随机采样隐向量𝒛~𝒑𝒛(∙)
    根据梯度下降算法更新 G 网络：
            𝛁𝝓𝔼𝒛~𝒑𝒛(∙)𝐥𝐨𝐠 (𝟏 − 𝑫𝜽(𝑮𝝓(𝒛)))
    end for
until 训练回合数达到要求
输出：训练好的生成器𝑮𝝓
"""