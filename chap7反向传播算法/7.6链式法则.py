# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/18 15:16
"""
前面我们介绍了输出层的梯度𝜕ℒ / ∂𝑤j𝑘计算方法，我们现在来介绍链式法则，它是能在不显式推导神经网络的数学表达式的情况下，
逐层推导梯度的核心公式，非常重要。
实际上，前面在推导梯度的过程中已经或多或少地用到了链式法则。考虑复合函数
𝑦 = 𝑓(𝑢)，𝑢 = 𝑔(𝑥)，则𝑑𝑦 / 𝑑𝑥可由𝑑𝑦 / 𝑑𝑢和𝑑𝑢 / 𝑑𝑥
推导出：
                    𝑑𝑦 / 𝑑𝑥 = 𝑑𝑦 / 𝑑𝑢 ∙ 𝑑𝑢 / 𝑑𝑥 = 𝑓′(𝑔(𝑥)) ∙ 𝑔′(𝑥)
考虑多元复合函数，𝑧 = 𝑓(𝑥, 𝑦)，其中𝑥 = 𝑔(𝑡), 𝑦 = ℎ(𝑡)，那么𝑑𝑧 / 𝑑𝑡的导数可以由𝜕𝑧 / 𝜕𝑥和𝜕𝑧 / 𝜕𝑦等推导出，具体表达为：
                    𝑑𝑧 / 𝑑𝑡 = 𝜕𝑧 / 𝜕𝑥 ∙ 𝑑𝑥 / 𝑑𝑡 + 𝜕𝑧 / 𝜕𝑦 ∙ 𝑑𝑦 / 𝑑t
例如，𝑧 = (2𝑡 + 1)2 + 𝑒𝑡2，令𝑥 = 2𝑡 + 1, 𝑦 = 𝑡2，则𝑧 = 𝑥2 + 𝑒𝑦，利用上式，可得：
                    𝑑𝑧 / 𝑑𝑡 = 𝜕𝑧 / 𝜕𝑥 ∙ 𝑑𝑥 / 𝑑𝑡 + 𝜕𝑧 / 𝜕𝑦 ∙ 𝑑𝑦 / 𝑑t = 2𝑥 ∙ 2 + 𝑒𝑦 ∙ 2𝑡
将𝑥 = 2𝑡 + 1, 𝑦 = 𝑡2代入可得：
                    𝑑𝑧 / 𝑑𝑡 = 2(2𝑡 + 1) ∙ 2 + 𝑒𝑡2 ∙ 2𝑡
即：
                    𝑑𝑧 / 𝑑𝑡 = 4(2𝑡 + 1) + 2𝑡𝑒𝑡2

神经网络的损失函数ℒ来自于各个输出节点𝑜𝑘(𝐾)，如下图 7.9 所示，其中输出节点𝑜𝑘(𝐾)又与隐藏层的输出节点𝑜j(J)相关联，
因此链式法则非常适合于神经网络的梯度推导。让我们来考虑损失函数ℒ如何应用链式法则。
前向传播时，数据经过𝑤𝑖j(J)传到倒数第二层的节点𝑜j(J)，再传播到输出层的节点𝑜𝑘(𝐾)。在每层只有一个节点时， 𝜕ℒ / 𝜕𝑤𝑖 (𝐽)可以利用链式法则，
逐层分解为：
            𝜕ℒ / 𝜕𝑤𝑖j(J) = 𝜕ℒ / 𝜕𝑜j(J) ∙ 𝜕𝑜j(J) / 𝜕𝑤𝑖j(J) = 𝜕ℒ / 𝜕𝑜𝑘(𝐾) ∙ 𝜕𝑜𝑘(𝐾) / 𝜕𝑜j(J) ∙ 𝜕𝑜j(J) / 𝜕𝑤𝑖j(J)

其中 𝜕ℒ / 𝜕𝑜𝑘(𝐾)可以由误差函数直接推导出，𝜕𝑜𝑘(𝐾) / 𝜕𝑜j(𝐽)可以由全连接层公式推导出，𝜕𝑜j(𝐽) / 𝜕𝑤𝑖j(𝐽)的导数即为输入𝑥𝑖(𝐼)。
可以看到，通过链式法则，我们不需要显式计算ℒ = 𝑓(𝑤𝑖j(J))的具体数学表达式，直接可以将偏导数进行分解，层层迭代即可推导出。

这里简单使用 TensorFlow 自动求导功能，来体验链式法则的魅力。例如：
import tensorflow as tf
# 构建待优化变量
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)
# 构建梯度记录器
with tf.GradientTape(persistent=True) as tape:
    # 非 tf.Variable 类型的张量需要人为设置记录梯度信息
    tape.watch([w1, b1, w2, b2])
    # 构建 2 层线性网络
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2
# 独立求解出各个偏导数
dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw1 = tape.gradient(y2, [w1])[0]
# 验证链式法则，2 个输出应相等
print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)

以上代码，通过自动求导功能计算出𝜕𝑦2 / 𝜕𝑦1、 𝜕𝑦1 / 𝜕𝑤1和𝜕𝑦2 / 𝜕𝑤1，借助链式法则我们可以推断𝜕𝑦2 / 𝜕𝑦1 ∙ 𝜕𝑦1 / 𝜕𝑤1 与𝜕𝑦2 / 𝜕𝑤1
应该是相等的，它们的计算结果如下：
tf.Tensor(2.0, shape=(), dtype=float32)
tf.Tensor(2.0, shape=(), dtype=float32)
可以看到𝜕𝑦2 / 𝜕𝑦1 ∙ 𝜕𝑦1 / 𝜕𝑤1 = 𝜕𝑦2 / 𝜕𝑤1，偏导数的传播是符合链式法则的。
"""