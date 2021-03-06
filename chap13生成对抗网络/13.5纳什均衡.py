# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/26 15:39
"""
现在我们从理论层面进行分析，通过博弈学习的训练方式，生成器 G 和判别器 D 分别会达到什么平衡状态。
具体地，我们将探索以下两个问题：
❑ 固定 G，D 会收敛到什么最优状态𝐷∗?
❑ 在 D 达到最优状态𝐷∗后，G 会收敛到什么状态？

首先我们通过𝒙𝑟~𝑝𝑟(∙)一维正态分布的例子给出一个直观的解释。如图 13.14 所示，黑色虚线曲线代表了真实数据的分布𝑝𝑟(∙)，为某正态分布𝒩(𝜇, 𝜎2)，
绿色实线代表了生成网络学习到的分布𝒙𝑓~𝑝𝑔(∙)，蓝色虚线代表了判别器的决策边界曲线，图 13.14 (a)、(b)、(c)、(d)分别代表了生成网络的学习轨迹。
在初始状态，如图 13.14(a)所示，𝑝𝑔(∙)分布与𝑝𝑟(∙)差异较大，判别器可以很轻松地学习到明确的决策边界，即图 13.14(a)中的蓝色虚线，
将来自𝑝𝑔(∙)的采样点判定为 0，𝑝𝑟(∙)中的采样点判定为 1。随着生成网络的分布𝑝𝑔(∙)越来越逼近真实分布𝑝𝑟(∙)，判别器越来越困难将真假样本区分开，如图 13.14(b)(c)所示。
最后，生成网络学习到的分布𝑝𝑔(∙) = 𝑝𝑟(∙)时，此时从生成网络中采样的样本非常逼真，判别器无法区分，即判定为真假样本的概率均等，如图 13.14(d)所示。
这个例子直观地解释了 GAN 网络的训练过程。
"""