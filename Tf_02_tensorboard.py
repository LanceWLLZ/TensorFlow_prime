# _*_ coding: utf-8 _*_
# 开发人员：LanceWLLZ
# 人员邮箱：723793008@qq.com
# 开发时间 2019/10/2918:39
# 文件名称 Tf_02_tensorboard
# 开发工具:PyCharm

# coding:utf-8
# tensorboard计算图的显示
#   cmd 输入tensorboard --logdir=D:\Data\Code_data\python_demo\TensorFlow_prime\tensorboar_log\01
#  用火狐登录该地址 http://DESKTOP-35562K8:6006 更新时直接刷新

import tensorflow as tf

x = 2
# 没有命名空间时
if (x==1):
    a = tf.add(1, 2)
    b = tf.multiply(a, 3)
    c = tf.add(a, b)
    d = tf.multiply(8, c)
    f = tf.div(d, 3)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tensorboar_log/01',sess.graph)
        print(sess.run(f))
        writer.close()

# 定义命名空间 input是常量
if (x==1):
    with tf.name_scope('input'):
            # fetch：就是同时运行多个op的意思
        input1 = tf.constant(3.0, name='A')  # 定义名称，会在tensorboard中代替显示
        input2 = tf.constant(4.0, name='B')
        input3 = tf.constant(5.0, name='C')
    with tf.name_scope('op'):
        # 加法
        add_01 = tf.add(input2, input3,name='add_01')
        # 乘法
        mul_01 = tf.multiply(input1, add_01,name='mul_01')
        # 除法 和被除数
        div_num = tf.Variable(initial_value=2.0,name='div_num',dtype=float)
        div_result = tf.divide(mul_01,div_num,name='div_result')
    with tf.Session() as sess:
        # 默认在当前py目录下的logs文件夹，没有会自己创建
        writer = tf.summary.FileWriter('tensorboar_log/01', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(div_num,3.0))
        print(sess.run(div_result))
        writer.close()

# 定义命名空间 input是变量
if (x==1):
    with tf.name_scope('input'):
            # fetch：就是同时运行多个op的意思
        input1 = tf.Variable(tf.constant(3.0),name='input01')
        input2 = tf.Variable(tf.constant(4.0),name='input02')
        input3 = tf.Variable(tf.constant(5.0),name='input03')
    with tf.name_scope('op'):
        # 加法
        add_01 = tf.add(input2, input3,name='add_01')
        # 乘法
        mul_01 = tf.multiply(input1, add_01,name='mul_01')
        # 除法 和被除数
        div_num = tf.Variable(initial_value=2.0,name='div_num',dtype=float)
        div_result = tf.divide(mul_01,div_num,name='div_result')
    with tf.Session() as sess:
        # 默认在当前py目录下的logs文件夹，没有会自己创建
        writer = tf.summary.FileWriter('tensorboar_log/01', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(div_num,3.0))
        print(sess.run(div_result))
        writer.close()

# 定义变量作用域
if (x==1):
    with tf.variable_scope('input'):
            # fetch：就是同时运行多个op的意思
        input1 = tf.get_variable(name='input01',shape=[1,2],initializer=tf.ones_initializer)
        input2 = tf.get_variable(name='input02',shape=[1,2],initializer=tf.ones_initializer)
        input3 = tf.get_variable(name='input03',shape=[1,2],initializer=tf.ones_initializer)
    with tf.name_scope('op'):
        # 加法
        add_01 = tf.add(input1, input2,name='add_01')
        # 乘法
        mul_01 = tf.multiply(input1, add_01,name='mul_01')
        # 除法 和被除数
        div_num = tf.get_variable(name='div_num',shape=[1,2],initializer=tf.ones_initializer)
        div_result = tf.divide(mul_01,div_num,name='div_result')
    with tf.Session() as sess:
        # 默认在当前py目录下的logs文件夹，没有会自己创建
        writer = tf.summary.FileWriter('tensorboar_log/01', sess.graph)
        sess.run(tf.global_variables_initializer())
        print(sess.run(div_result))
        writer.close()


# tensorboard --logdir=D:\Data\Code_data\python_demo\TensorFlow_prime\tensorboar_log\01
#   http://DESKTOP-35562K8:6006



