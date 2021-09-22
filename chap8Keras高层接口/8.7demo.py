# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/22 10:31
import tensorflow as tf
import tensorboard
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default(): # 写入环境
    # 当前时间戳 step 上的数据为 loss，写入到名为 train-loss 数据库中
    tf.summary.scalar('train-loss', float(loss), step=step)
