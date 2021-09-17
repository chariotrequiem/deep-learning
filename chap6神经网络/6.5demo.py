# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/17 15:13
import tensorflow as tf
from tensorflow import keras

x = tf.constant([2., 1., 0.1])
y = tf.nn.softmax(x)
print(y)

print('----------------categorical_crossentropy接口函数，将Softmax与交叉熵损失函数同时实现------------------')
z = tf.random.normal([2, 10])  # 构造输出层的输出
y_onehot = tf.constant([1, 3])  # 构造真实值
# print(y_onehot)
y_onehot = tf.one_hot(y_onehot, depth=10)  # one-hot编码
# 输出层未使用Softmax函数，故from_logits设置为True
# 这样categorical_crossentropy函数在计算损失函数前，会先调用Softmax函数
loss = keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
loss = tf.reduce_mean(loss)  # 计算平均交叉熵损失
print(loss)


print('--------------------------除了函数接口，也可以利用 losses.CategoricalCrossentropy(from_logits)类方式同时实现'
      'Softmax与交叉熵损失函数的计算，from_logits 参数的设置方式相同。--------------------------')
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot, z)  # 计算损失(不必使用reduce_mean来计算平均交叉熵损失)
print(loss)


x = tf.linspace(-6., 6., 10)
y = tf.nn.tanh(x)
print(y)
