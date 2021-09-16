# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/15 16:10
import tensorflow as tf
from tensorflow import keras
a = tf.constant([1, 2, 3, 4, 5, 6])  # 第一个句子
b = tf.constant([7, 8, 1, 6])  # 第二个句子
b = tf.pad(b, [[0, 2]])  # 句子末尾填充2个0
# print(b)

# 填充后的句子张量一致，再将这2句子Stack在一起，代码如下：
c = tf.stack([a, b], axis=0)
print(c)


total_words = 10000  # 设定词汇量大小
max_review_len = 80  # 最大句子长度
embedding_len = 100  # 词向量长度
# 加载IMDB数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# 将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len, truncating='post', padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len, truncating='post', padding='post')

print(x_train.shape, x_test.shape)  # 打印等长的句子张量形状


print('-------------practice--------------')
a = tf.constant([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10]])
b = tf.constant([[2, 3, 4], [8, 9, 10]])
b = tf.pad(b, [[0, 0], [2, 0]])
# print(b)
c = tf.stack([a, b], axis=0)
print(c)

print('----------------------------------')
a = tf.random.normal([2, 2, 2, 2])
b = tf.pad(a, [[0, 0], [1, 1], [1, 1], [0, 0]])
print(b)