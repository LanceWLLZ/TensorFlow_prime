# _*_ coding: utf-8 _*_
# 开发人员：LanceWLLZ
# 人员邮箱：723793008@qq.com
# 开发时间 2019/10/2918:39
# 文件名称 Tf_02_tensorboard
# 开发工具:PyCharm

# https://www.cnblogs.com/lyc-seu/p/8647792.html
# 关于tf.summary 的相关用法
# 训练过程数据收集 直方图 分布图
'''
    tf.summary.scalar用来显示标量信息，其格式为：
    tf.summary.scalar(tags, values, collections=None, name=None)

    tf.summary.histogram
    用来显示直方图信息，其格式为：
    tf.summary.histogram(tags, values, collections=None, name=None)

    tf.summary.distribution
    分布图，一般用于显示weights分布

    tf.summary.FileWriter
    指定一个文件用来保存图。
    格式：tf.summary.FileWritter(path,sess.graph)

    tf.summary.merge
    tf.summary.merge_all
    merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
    tf.summary.merge(inputs, collections=None, name=None) 保存特定的变量值

    过程数据记录的用法
    writer = tf.summary.FileWriter(filename, sess.graph)
    summary_merged = tf.summary.merge_all（）
    with tf.Session as sess:
        for i in range(1000)
            .......(运行网络和逆向传播)
            summOut = sess.run([summary_merged] # 保存每次运行的结果
            writer.add_summary(summOut, i)
'''

import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2 as cv
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from tensorflow.python.platform import gfile

print(tf.__version__)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
# 搭建一个隐层神经元数量为30的 只有两层的NN
num_hiddens = 30
with tf.name_scope("input") as scope:
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="x_input")

with tf.name_scope("output") as scope:
    y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="y_output")

with tf.name_scope("hidden") as scope:
    w1 = tf.Variable(tf.truncated_normal(shape=[784, num_hiddens]), name="w1")
    b1 = tf.Variable(tf.truncated_normal(shape=[1, num_hiddens]), name="b1")
    tf.summary.histogram("weight_1", w1)  # 直方图信息tf.summary.histogram(tags, values, collections=None, name=None)
    nn1 = tf.add(tf.matmul(x, w1), b1)
    h1 = tf.sigmoid(nn1, name="hActivation")

with tf.name_scope("output_layer") as scope:
    w2 = tf.Variable(tf.truncated_normal(shape=[num_hiddens, 10]), name="w2")
    b2 = tf.Variable(tf.truncated_normal(shape=[1, 10]), name="b2")
    tf.summary.histogram("weight_2", w2)
    nn2 = tf.add(tf.matmul(h1, w2), b2)
    out = tf.sigmoid(nn2, name="outActivation")

with tf.name_scope("lost_potimizer") as scope:
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=nn2, labels=y)
    loss = tf.reduce_sum(tf.square(tf.subtract(y, out)))
    tf.summary.scalar("cost", loss)     # 标量信息 tf.summary.scalar(tags, values, collections=None, name=None)

    step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.name_scope("accu") as scope:
    acc_mat = tf.equal(tf.arg_max(out, 1), tf.argmax(y, 1))
    acc_ret = tf.reduce_sum(tf.cast(acc_mat, dtype=tf.float32))
    tf.summary.scalar("accuracy", acc_ret)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_merged = tf.summary.merge_all()
    filename = "tensorboar_log/02/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = tf.summary.FileWriter(filename, sess.graph)
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        curr_, summOut = sess.run([step, summary_merged], feed_dict={x: batch_xs, y: batch_ys})
        writer.add_summary(summOut, i)
        if (i + 1) % 1000 == 0:
            curr_acc = sess.run(acc_ret, feed_dict={x: mnist.test.images[:1000], y: mnist.test.labels[:1000]})
            print("current acc : %f" % (curr_acc))

# tensorboard --logdir=D:\Data\Code_data\python_demo\TensorFlow_prime\tensorboar_log\02\2019-10-30-19-56-30
#  用火狐登录该地址 http://DESKTOP-35562K8:6006 更新时直接刷新