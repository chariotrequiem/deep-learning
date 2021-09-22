# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 21:42
import tensorflow as tf
from tensorflow.keras import datasets


def preprocess(x, y):
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    # x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.one_hot(y, depth=10)  # one_hot编码
    return x, y


def db(x, y, shuffle_nums, batch_nums):
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.shuffle(shuffle_nums).batch(batch_nums).map(preprocess)
    return db


# 构建训练集
"""train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建Dataset对象
train_db = train_db.shuffle(60000)  # 将对象随机打散
train_db = train_db.batch(128)  # 批训练，每批训练128张图片
train_db = train_db.map(preprocess)  # 进行预处理
# 构建测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000).batch(128).map(preprocess)"""

if __name__ == '__main__':
    # 加载mnist数据集
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    print(x.shape, y.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    train_db = db(x, y, 60000, 128)
    test_db = db(x_test, y_test, 10000, 128)
    print(train_db, test_db)



