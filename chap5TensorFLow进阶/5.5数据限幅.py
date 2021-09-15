# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 16:41
"""
考虑怎么实现非线性激活函数 ReLU 的问题。它其实可以通过简单的数据限幅运算实现，限制元素的范围𝑥 ∈ [0, +∞)即可。
在 TensorFlow 中，可以通过 tf.maximum(x, a)实现数据的下限幅，即𝑥 ∈ [𝑎, +∞)；
可以通过 tf.minimum(x, a)实现数据的上限幅，即𝑥 ∈ (−∞,𝑎]，
举例如下：
In [33]:x = tf.range(9)
tf.maximum(x,2) # 下限幅到 2
Out[33]:<tf.Tensor: id=48, shape=(9,), dtype=int32, numpy=array([2, 2, 2, 3, 4, 5, 6, 7, 8])>
In [34]:tf.minimum(x,7) # 上限幅到 7
Out[34]:<tf.Tensor: id=41, shape=(9,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 7])>

基于 tf.maximum 函数，我们可以实现 ReLU 函数如下：
def relu(x): # ReLU 函数
    return tf.maximum(x,0.) # 下限幅为 0 即可
通过组合 tf.maximum(x, a)和 tf.minimum(x, b)可以实现同时对数据的上下边界限幅，即𝑥 ∈ [𝑎, 𝑏]，例如:
In [35]:x = tf.range(9)
tf.minimum(tf.maximum(x,2),7) # 限幅为 2~7
Out[35]:<tf.Tensor: id=57, shape=(9,), dtype=int32, numpy=array([2, 2, 2, 3, 4, 5, 6, 7, 7])>
更方便地，我们可以使用 tf.clip_by_value 函数实现上下限幅：
In [36]:x = tf.range(9)
tf.clip_by_value(x,2,7) # 限幅为 2~7
Out[36]:<tf.Tensor: id=66, shape=(9,), dtype=int32, numpy=array([2, 2, 2, 3, 4, 5, 6, 7, 7])>
"""