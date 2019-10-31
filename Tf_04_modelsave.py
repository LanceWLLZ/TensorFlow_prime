# _*_ coding: utf-8 _*_
# 开发人员:LanceWLLZ
# 人员邮箱:723793008@qq.com
# 开发时间 2019/10/3020:35
# 文件名称 Tf_04_modelsave
# 开发工具:PyCharm

# 训练模型后保存模型
# 训练模型后保存参数
# 使用训练好的模型(重新加载模型)
# 模型调用
# 加载网络
# 加载张量
'''
主要有save restore方法 和
'''
'''
1、保存训练后的模型参数
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())#一定要先初始化整个流
        #在这里训练网络
        ... 
        #保存参数
        saver = tf.train.Saver()
            saver.save(sess,PATH)#PATH就是要保存的路径  
2、保存模型
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())#一定要先初始化整个流
        #在这里训练网络
        ... 
        #保存参数
    builder = tf.saved_model.builder.SaveModelBuilder(PATH)#PATH是保存路径
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.TRAINING])#保存整张网络及其变量,
                                                                                        #这种方法是可以保存多张网络的
    builder.save()#完成保存

3、模型调用
    用tf.saved_model.builder 保存的模型同样用该语句调用
    with tf.Session()as sess:
         tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.TRAINING],PATH)#PATH还是路径
'''
import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2 as cv
import time
import os
import operator
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from tensorflow.python.platform import gfile

# 整个网络结构:卷积 - >池化 - >卷积 - >池化 - >全连接 - > dropout->全连接 - > SOFTMAX
# define占位符
#
print(tf.__version__)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
x = 2
if (x == 1):
    # define占位符
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32,name='x')
    y = tf.placeholder(shape=[None, 10], dtype=tf.float32,name='y')
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convolution layer 1
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.truncated_normal(shape=[32], stddev=0.1))
    conv1_out = tf.nn.conv2d(input=x_image, filter=conv1_w, strides=[1, 1, 1, 1], padding='SAME')
    conv1_relu = tf.nn.relu(tf.add(conv1_out, conv1_bias))
    # conv1_relu = tf.nn.sigmoid(tf.add(conv1_out, conv1_bias))

    # max pooling 1
    maxpooling_1 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # convolution layer 2
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1))
    conv2_out = tf.nn.conv2d(input=maxpooling_1, filter=conv2_w, strides=[1, 1, 1, 1], padding='SAME')
    conv2_relu = tf.nn.relu(tf.add(conv2_out, conv2_bias))
    # conv2_relu = tf.nn.sigmoid(tf.add(conv2_out, conv2_bias))

    # max pooling 2
    maxpooling_2 = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fc-1
    w_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1, dtype=tf.float32))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2 = tf.reshape(maxpooling_2, [-1, 7 * 7 * 64])
    output_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2, w_fc1), b_fc1))

    # drop_out
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    h1 = tf.nn.dropout(tf.nn.sigmoid(output_fc1), keep_prob=keep_prob)

    # fc-2
    w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1, dtype=tf.float32))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    y_conv = tf.add(tf.matmul(h1, w_fc2), b_fc2)

    # 损失函数设定
    cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y)
    loss = tf.reduce_mean(cross_loss)
    # 训练器设定
    step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    # 准确率观察
    acc_mat = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(y, 1))
    acc_ret = tf.reduce_sum(tf.cast(acc_mat, dtype=tf.float32),name='acc_ret')

    saver = tf.train.Saver()# 位于该句之后的变量不会被保存
    ''' var_list 默认为空，保存所有变量
    max_to_keep：保存多少个最新的checkpoint文件，默认为5，即保存最近五个checkpoint文件
    save_relative_paths：为True时，checkpoint文件将不会记录完整的模型路径，而只会仅仅记录模型名字，
    这方便于将保存下来的模型复制到其他目录并使用的情况
    '''
    model_filename = "./tensorboar_log/04/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/test_model"
    prediction = tf.arg_max(y_conv, 1)
    start = time.time()

    # 运行sess 训练模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # summary_merged = tf.summary.merge_all()
        filename = "tensorboar_log/04/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"log"
        # writer = tf.summary.FileWriter(filename, sess.graph)
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            curr_= sess.run(step,feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            # curr_, summOut = sess.run([step, summary_merged], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            # writer.add_summary(summOut, i)
            if (i + 1) % 1000 == 0:
                curr_acc = sess.run(acc_ret,
                                    feed_dict={x: mnist.test.images[:1000], y: mnist.test.labels[:1000],
                                               keep_prob: 1.0})
                print("current acc : %f" % (curr_acc))
        saver.save(sess, save_path=model_filename, global_step=i+1)

    end = time.time()
    print('counting time', end - start)
    # model_filename = "./tensorboar_log/04/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/test.model"
    # model_filename ='testmodel'
    # saver.save(sess, save_path=model_filename, global_step=10000)

if(x==2):
    # 从检查点模型恢复
    model_meta_filename =r"D:\Data\Code_data\python_demo\TensorFlow_prime\tensorboar_log\04\2019-10-31-15-15-48"

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta_filename+"/test_model-10000.meta") # 恢复meta图结构
        saver.restore(sess, tf.train.latest_checkpoint(model_meta_filename)) # 从检查点恢复节点参数 给变量赋值
        graph = tf.get_default_graph()# 获取图结构
        x = graph.get_tensor_by_name("x:0")# 获取name相关节点
        y = graph.get_tensor_by_name("y:0")
        acc_ret = graph.get_tensor_by_name("acc_ret:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        curr_acc = sess.run(acc_ret,
                            feed_dict={x: mnist.test.images[:1000], y: mnist.test.labels[:1000],
                                       keep_prob: 1.0})
        print("current acc : %f" % (curr_acc))
