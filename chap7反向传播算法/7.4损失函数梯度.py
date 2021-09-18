# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 10:48
"""
前面已经介绍了常见的损失函数，这里主要推导均方误差损失函数和交叉熵损失函数的梯度表达式。

7.4.1均方误差函数梯度
均方误差损失函数表达式为：
                        ℒ = 1 / 2∑(𝑦𝑘 − 𝑜𝑘)2
上式中的1/2项用于简化计算，也可以利用1/𝐾进行平均，这些缩放运算均不会改变梯度方向。
则它的偏导数∂ℒ / ∂𝑜𝑖可以展开为：
                        ∂ℒ / ∂𝑜𝑖 = 1 / 2∑∂(𝑦𝑘 − 𝑜𝑘)2 / ∂𝑜𝑖
利用复合函数导数法则分解为：
                        𝜕ℒ / 𝜕𝑜𝑖 = 1 / 2∑2 ∙ (𝑦𝑘 − 𝑜𝑘) ∙ 𝜕(𝑦𝑘 − 𝑜𝑘) / 𝜕𝑜𝑖
即 ：
                        𝜕ℒ / 𝜕𝑜𝑖 = ∑(𝑦𝑘 − 𝑜𝑘) ∙ −1 ∙ 𝜕𝑜𝑘 / 𝜕𝑜𝑖
                                    = ∑(𝑜𝑘 − 𝑦𝑘)∙ 𝜕𝑜𝑘 / 𝜕𝑜𝑖
考虑到𝜕𝑜𝑘 / 𝜕𝑜𝑖仅当𝑘 = 𝑖时才为 1，其它点都为 0，也就是说，偏导数𝜕𝜕𝑜ℒ𝑖
只与第𝑖号节点相关，与
其它节点无关，因此上式中的求和符号可以去掉。均方误差函数的导数可以推导为：
                        𝜕ℒ / 𝜕𝑜𝑖 = (𝑜𝑖 − 𝑦𝑖)


7.4.2交叉熵函数梯度
在计算交叉熵损失函数时，一般将 Softmax 函数与交叉熵函数统一实现。我们先推导Softmax 函数的梯度，再推导交叉熵函数的梯度。
Softmax 梯度
回顾 Softmax 函数的表达式：
                        𝑝𝑖 = 𝑒^𝑧𝑖 / ∑ 𝑒^𝑧𝑘
它的功能是将𝐾个输出节点的值转换为概率，并保证概率之和为 1，如图 7.6 所示。
回顾
                        𝑓(𝑥) = 𝑔(𝑥) / ℎ(𝑥)
函数的导数表达式：
                        𝑓′(𝑥) = 𝑔′(𝑥)ℎ(𝑥) − ℎ′(𝑥)𝑔(𝑥) / ℎ(𝑥)2
对于 Softmax 函数，𝑔(𝑥) = 𝑒^𝑧𝑖，ℎ(𝑥) = ∑ 𝑒^𝑧𝑘  ，下面我们根据𝑖 = 𝑗和𝑖 ≠ 𝑗来分别推导Softmax 函数的梯度。

❑ 𝑖 = 𝑗时。Softmax 函数的偏导数𝜕𝑝𝑖 / 𝜕𝑧可以展开为：
                        𝜕𝑝𝑖 / 𝜕𝑧 = 𝜕 𝑒^𝑧𝑖 / ∑ 𝑒𝑧𝑘 / 𝜕𝑧j = 𝑒𝑧𝑖 ∑ 𝑒^𝑧𝑘 − 𝑒𝑧j 𝑒𝑧𝑖 / (∑𝑒𝑧𝑘 )2
提取公共项𝑒𝑧𝑖：
                            = 𝑒𝑧𝑖(∑ 𝑒𝑧𝑘 − 𝑒𝑧j ) / (∑ 𝑒𝑧𝑘)2
拆分为两部分：
                            = 𝑒^𝑧𝑖 / ∑𝑒𝑧𝑘  × (∑ 𝑒^𝑧𝑘 − 𝑒^𝑧j ) / ∑ 𝑒^𝑧𝑘
可以看到，上式是概率值𝑝𝑖和1 − 𝑝j的相乘，同时满足𝑝𝑖 = 𝑝j。因此𝑖 = 𝑗时，Softmax 函数的偏导数𝜕𝑝𝑖 / 𝜕𝑧j为：
                            𝜕𝑝𝑖 / 𝜕𝑧j = 𝑝𝑖(1 − 𝑝j),𝑖 = j

❑ 𝑖 ≠ 𝑗时。展开 Softmax 函数为:
                            𝜕𝑝𝑖 / 𝜕𝑧j = 𝜕 𝑒𝑧𝑖 / ∑ 𝑒𝑧𝑘 / 𝜕𝑧j = 0 − 𝑒𝑧j 𝑒𝑧𝑖 / (∑ 𝑒^𝑧𝑘 )2
去掉 0 项，并分解为两项相乘：
                            = −𝑒^𝑧j / ∑ 𝑒𝑧𝑘  × 𝑒𝑧𝑖 / ∑ 𝑒𝑧𝑘
即：
                            𝜕𝑝𝑖 / 𝜕𝑧j = −𝑝j ⋅ 𝑝𝑖

可以看到，虽然 Softmax 函数的梯度推导过程稍复杂，但是最终表达式还是很简洁的，偏导数表达式如下：
                                𝜕𝑝𝑖 / 𝜕𝑧 = { 𝑝𝑖(1 − 𝑝𝑗) 当𝑖 = 𝑗
                                            −𝑝𝑖 ⋅ 𝑝𝑗     当𝑖 ≠ 𝑗

交叉熵梯度
考虑交叉熵损失函数的表达式： ℒ = −∑𝑦𝑘 log(𝑝𝑘)
这里直接来推导最终损失值ℒ对网络输出 logits 变量𝑧𝑖的偏导数，展开为：
                                𝜕ℒ / 𝜕𝑧𝑖 = −∑𝑦𝑘 𝜕 log(𝑝𝑘) / 𝜕𝑧𝑖
将log ℎ复合函数分解为：
                                = −∑𝑦𝑘 𝜕 log(𝑝𝑘) / 𝜕𝑝𝑘 ∙ 𝜕𝑝𝑘 / 𝜕𝑧𝑖
即                              = −∑𝑦𝑘 1 / 𝑝𝑘 ∙ 𝜕𝑝𝑘 / 𝜕𝑧𝑖
其中𝜕𝑝𝑘 / 𝜕z𝑖即为我们已经推导的 Softmax 函数的偏导数。

将求和符号拆分为𝑘 = 𝑖以及𝑘 ≠ 𝑖的两种情况，并代入𝜕𝑝𝑘 / 𝜕z𝑖求解的公式，可得
                    𝜕ℒ / 𝜕𝑧𝑖 = −𝑦𝑖(1 − 𝑝𝑖) −∑𝑘≠𝑖 𝑦𝑘 ⋅ 1 / 𝑝𝑘 (−𝑝𝑘 ⋅ 𝑝𝑖)
进一步化简为
                    = −𝑦𝑖(1 − 𝑝𝑖) +∑𝑘≠𝑖 𝑦𝑘 ⋅ 𝑝𝑖
                    = −𝑦𝑖 + 𝑦𝑖𝑝𝑖 +∑𝑘≠𝑖 𝑦𝑘 ⋅ 𝑝𝑖
提供公共项𝑝𝑖，可得：
                    𝜕ℒ / 𝜕𝑧𝑖 = 𝑝𝑖 (𝑦𝑖 +∑𝑘≠𝑖 𝑦𝑘) − 𝑦𝑖
完成交叉熵函数的梯度推导。
特别地，对于分类问题中标签𝑦通过 One-hot 编码的方式，则有如下关系：
                        ∑𝑘 𝑦𝑘 = 1
                        𝑦𝑖 +∑𝑘≠𝑖 𝑦𝑘 = 1
因此交叉熵的偏导数可以进一步简化为：𝜕ℒ / 𝜕𝑧𝑖 = 𝑝𝑖 − 𝑦𝑖
"""