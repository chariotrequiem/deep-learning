# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 17:03
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
a = tf.range(8)
a = tf.reshape(a, [4, 2])
print(a)

b = tf.gather(a, [3, 1, 0, 2], axis=0)
print(b)


# 如果希望抽查第[2,3]班级的第[3,4,6,27]号同学的科目成绩，则可以通过组合多个 tf.gather 实现
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量（4个班级，35名同学，8门科目）
students = tf.gather(x, [1, 2], axis=0)  # 首先抽取第2，3班级
scores = tf.gather(students, [2, 3, 5, 26], axis=1)
print(scores)


# 通过 tf.gather_nd 函数，可以通过指定每次采样点的多维坐标来实现采样多个点的目的。
score_0 = tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]])
print(score_0)

score_1 = tf.gather_nd(x, [[1, 1, 2], [2, 2, 3], [3, 3, 4]])
print(score_1)

classes = tf.boolean_mask(x, mask=[True, False, True, False], axis=0)
print(classes)

counts = tf.boolean_mask(x, mask=[True, False, False, True, True, False, False, True], axis=2)
print(counts)

x = tf.random.uniform([2, 3, 8], maxval=100, dtype=tf.int32)
print(tf.boolean_mask(x, [[True, True, False], [False, True, True]]))


print('-----------------------------------')
a = tf.ones([3, 3])
b = tf.zeros([3, 3])
# 构造采样条件
cond = tf.constant([[True, False, True], [True, False, False], [True, True, True]])
c = tf.where(cond, a, b)
print(c)
d = tf.where(cond)
print(d)

print('------------------------------------')
x = tf.random.normal([3, 3])
print(x)
# 通过比较运算，得到所有正数的掩码
mask = x > 0
print(mask)
# 通过tf.where提取掩码True元素的索引坐标
indices = tf.where(mask)  # 提取所有大于0的元素序列
print(indices)
# 拿到索引后，通过tf.gather_nd即可恢复所有正数的元素
positive = tf.gather_nd(x, indices)
print(positive)
# 得到掩码后，也可以直接通过tf.boolean_mask获取所有正数的元素向量
positive = tf.boolean_mask(x, mask)
print(positive)

print('-----------------------------------')
# 构造需要刷新数据的参数位置，即为4、3、1和7号位置
indices = tf.constant([[4], [3], [1], [7]])
print(indices.shape)
updates = tf.constant([4.4, 3.3, 1.1, 7.7])
print(updates.shape)
a = tf.scatter_nd(indices, updates, [8])
print(a)

print('------------------------------------')
indices = tf.constant([[1], [3]])
updates = tf.constant([
    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
])
# 在shape为[4, 4, 4]的白板上根据indices写入updates
c = tf.scatter_nd(indices, updates, [4, 4, 4])
print(c)

"""print('--------------------------------------')
x = tf.linspace(-8, 8, 100)
y = tf.linspace(-8, 8, 100)
print(x)
print(y)"""


print('-----------tf.gather----------------')
a = tf.random.uniform([4, 35, 8], minval=15, maxval=100, dtype=tf.int32)
b = tf.gather(a, [0, 3, 8, 11, 12, 26], axis=1)
print(b)

# 收集所有学生的第3，5门成绩
c = tf.gather(a, [4, 6], axis=2)
print(c)

# 搜集第[2,3]班级的第[3,4,6,27]号同学的科目成绩，
d = tf.gather(a, [1, 2], axis=0)
d = tf.gather(d, [2, 3, 5, 26], axis=1)
print(d)

# tf.gather_nd()函数
d = tf.gather_nd(a, [[[1, 2], [1, 3], [1, 5], [1, 26]], [[2, 2], [2, 3], [2, 5], [2, 26]]])
print(d)


print('-------------tf.boolean_mask()函数--------------')
x = tf.random.uniform([2, 3, 8], maxval=100, dtype=tf.int32)
mask = [[True, True, False], [False, True, True]]
a = tf.boolean_mask(x, mask)
b = tf.gather_nd(x, [[0, 0], [0, 1], [1, 1], [1, 2]])
print(a)
print(b)


print('-----------------tf.where()函数------------------')
x = tf.random.normal([4, 4])
print(x)
mask = x > 0
indices = tf.where(mask)
print(indices)
c = tf.gather_nd(x, indices)
print(c)