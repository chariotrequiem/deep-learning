# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/25 16:29
"""
我们用一个漫画家的成长轨迹来形象介绍生成对抗网络的思想。考虑一对双胞胎兄弟，分别称为老二 G 和老大 D，G 学习如何绘制漫画，
D 学习如何鉴赏画作。还在娃娃时代的两兄弟，尚且只学会了如何使用画笔和纸张，G 绘制了一张不明所以的画作，如图13.2(a)所示，
由于此时 D 鉴别能力不高，觉得 G 的作品还行，但是人物主体不够鲜明。在 D 的指引和鼓励下，G 开始尝试学习如何绘制主体轮廓和使用简单的色彩搭配。

一年后，G 提升了绘画的基本功，D 也通过分析名作和初学者 G 的作品，初步掌握了鉴别作品的能力。此时 D 觉得 G 的作品人物主体有了，
如图 13.2(b)，但是色彩的运用还不够成熟。数年后，G 的绘画基本功已经很扎实了，可以轻松绘制出主体鲜明、颜色搭配合适和逼真度较高的画作，
如图 13.2(c)，但是 D 同样通过观察 G 和其它名作的差别，提升了画作鉴别能力，觉得 G 的画作技艺已经趋于成熟，但是对生活的观察尚且不够，
作品没有传达神情且部分细节不够完美。又过了数年，G 的绘画功力达到了炉火纯青的地步，绘制的作品细节完美、风格迥异、惟妙惟肖，
宛如大师级水准，如图 13.2(d)，即便此时的D 鉴别功力也相当出色，亦很难将 G 和其他大师级的作品区分开来。

上述画家的成长历程其实是一个生活中普遍存在的学习过程，通过双方的博弈学习，相互提高，最终达到一个平衡点。GAN 网络借鉴了博弈学习的思想，
分别设立了两个子网络：负责生成样本的生成器 G 和负责鉴别真伪的鉴别器 D。类比到画家的例子，生成器 G就是老二，鉴别器 D 就是老大。
鉴别器 D 通过观察真实的样本和生成器 G 产生的样本之间的区别，学会如何鉴别真假，其中真实的样本为真，生成器 G 产生的样本为假。
而生成器 G 同样也在学习，它希望产生的样本能够获得鉴别器 D 的认可，即在鉴别器 D 中鉴别为真，因此生成器 G 通过优化自身的参数，
尝试使得自己产生的样本在鉴别器 D 中判别为真。生成器 G 和鉴别器 D 相互博弈，共同提升，直至达到平衡点。此时生成器 G 生成的样本非常逼真，使得鉴别器 D 真假难分。

在原始的 GAN 论文中，Ian Goodfellow 使用了另一个形象的比喻来介绍 GAN 模型：生成器网络 G 的功能就是产生一系列非常逼真的假钞试图欺骗鉴别器 D，
而鉴别器 D 通过学习真钞和生成器 G 生成的假钞来掌握钞票的鉴别方法。这两个网络在相互博弈的过程中间同步提升，直到生成器 G 产生的假钞非常的逼真，
连鉴别器 D 都真假难辨。

这种博弈学习的思想使得 GAN 的网络结构和训练过程与之前的网络模型略有不同，下面我们来详细介绍 GAN 的网络结构和算法原理。
"""