# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 15:01
"""
为了计算分类任务的准确率等指标，一般需要将预测结果与真实标签比较，统计比较结果中正确的数量来计算准确率。
考虑100个样本的预测结果，通过tf.argmax获取预测类别，实现如下：
In [24]:out = tf.random.normal([100,10])
out = tf.nn.softmax(out, axis=1) # 输出转换为概率
pred = tf.argmax(out, axis=1) # 计算预测值
Out[24]:<tf.Tensor: id=272, shape=(100,), dtype=int64, numpy=
array([0, 6, 4, 3, 6, 8, 6, 3, 7, 9, 5, 7, 3, 7, 1, 5, 6, 1, 2, 9, 0, 6,
       5, 4, 9, 5, 6, 4, 6, 0, 8, 4, 7, 3, 4, 7, 4, 1, 2, 4, 9, 4,…

变量pred保存了这100个样本的预测类别值，我们将这100个样本的真实标签比较，例如：
In [25]: # 模型生成真实标签
y = tf.random.uniform([100],dtype=tf.int64,maxval=10)
Out[25]:<tf.Tensor: id=281, shape=(100,), dtype=int64, numpy=
array([0, 9, 8, 4, 9, 7, 2, 7, 6, 7, 3, 4, 2, 6, 5, 0, 9, 4, 5, 8, 4, 2,
 5, 5, 5, 3, 8, 5, 2, 0, 3, 6, 0, 7, 1, 1, 7, 0, 6, 1, 2, 1, 3, …

即可获得代表每个样本是否预测正确的布尔类型张量。通过 tf.equal(a, b)(或 tf.math.equal(a, b)，两者等价)函数可以比较
这 2 个张量是否相等，例如：
In [26]:out = tf.equal(pred,y) # 预测值与真实值比较，返回布尔类型的张量
Out[26]:<tf.Tensor: id=288, shape=(100,), dtype=bool, numpy=
array([False, False, False, False, True, False, False, False, False,
 False, False, False, False, False, True, False, False, True,…

tf.equal()函数返回布尔类型的张量比较结果，只需要统计张量中 True 元素的个数，即可知道预测正确的个数。
为了达到这个目的，我们先将布尔类型转换为整形张量，即 True 对应为 1，False 对应为 0，再求和其中 1 的个数，
就可以得到比较结果中 True 元素的个数：
In [27]:out = tf.cast(out, dtype=tf.float32) # 布尔型转 int 型
correct = tf.reduce_sum(out) # 统计 True 的个数
Out[27]:<tf.Tensor: id=293, shape=(), dtype=float32, numpy=12.0>
可以看到，我们随机产生的预测数据中预测正确的个数是 12，因此它的准确度是
                    accuracy =12/100 = 12%
这也是随机预测模型的正常水平。

除了比较相等的tf.equal(a, b)函数，其他的比较函数用法类似，如下表所示：
                       常用比较函数总结
            函数                           比较逻辑
            tf.math.greater                a > b
            tf.math.less                   a < b
            tf.math.greater_equal          a ≥ b
            tf.math.less_equal             a ≤ b
            tf.math.not_equal              a ≠ b
            tf.math.is_nan                 a = nan
"""