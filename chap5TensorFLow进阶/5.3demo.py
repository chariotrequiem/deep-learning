# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 15:23
import tensorflow as tf
out = tf.random.normal([100, 10])
out = tf.nn.softmax(out, axis=1)  # 输出转换为概率
# print(out)
pred = tf.argmax(out, axis=1)  # 计算预测值
# print(pred)

# 模拟生成真实标签
y = tf.random.uniform([100], dtype=tf.int64, maxval=10)
# print(y)
out = tf.equal(pred, y)  # 比较两个张量是否相等
# print(out)
# 将布尔类型转换为int型
out = tf.cast(out, dtype=tf.int32)
print(out)
correct = tf.reduce_sum(out)  # 统计True的个数
acc = correct/out.shape[0]
print('正确预测的成功率为{}%'.format(acc*100))