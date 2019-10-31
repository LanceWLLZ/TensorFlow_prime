# _*_ coding: utf-8 _*_
# 开发人员：LanceWLLZ
# 人员邮箱：723793008@qq.com
# 开发时间 2019/10/2916:33
# 文件名称 Tf_variable_01
# 开发工具:PyCharm

import tensorflow as tf

x = 2

#   01  tf.Variable
'''
    重要的参数包括initial_value，shape，dtype，name
    不指定name时以程序中的变量名作为name
    name重复使用会自动在name后面加上 _n
'''
if (x == 1):
    a = tf.Variable(tf.constant(1.0, shape=[1]))
    b = tf.Variable(tf.constant(1.0, shape=[1]))
    c = tf.Variable(tf.constant(1.0, shape=[1]))
    d = tf.Variable(initial_value=3, name='d')
    e = tf.Variable(initial_value=3, name='d')
    print(a)  # 不指定name 则会以Variable作为name的起始  Variable:0
    print(b)  # Variable_1:0   已经有Variable:0的情况下，多加上“_1:0”
    print(c)  # Variable_2:0
    print(d)  # d:0
    print(e)  # d_1:0

#   02  tf.get_variable
'''
    重要的参数包括 shape name dtype initializer 和tf.Variable的initial_value 一样
    # tf.get_variable 和tf.Variable
    # tf.get_variable与tf.Variable最大不同就是tf.get_variable中name参数是一个必填的参数
    # tf.get_variable的name参数是不能重名的
    
    tf.get_variable（）函数中的 的initializer参数的选择如下
    tf.constant_initializer：常量初始化函数
    tf.random_normal_initializer：正态分布
    tf.truncated_normal_initializer：截取的正态分布
    tf.random_uniform_initializer：均匀分布
    tf.zeros_initializer：全部是0
    tf.ones_initializer：全是1
    tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
'''
if (x == 1):
    a = tf.get_variable(name='a1', shape=[1, 2], initializer=tf.random_normal_initializer)
    # b = tf.get_variable(name='a1', shape=[1, 2]) # 名字相同会报错
    print(a)


#   03  命名空间 变量名的圈定 tf.name_scope
'''
    作用：对变量的name属性进行圈定，变量name前都会有name_scope的信息
    便于在tensorboard中展示清晰的逻辑关系图，但这个不对变量本身进行限制
    好比甲、乙、丙、丁属于陈家，这里“陈家”就是一个name_scope划定的区域，虽然他们只属于陈家，
    但他们依然可以去全世界的任何地方，并不会只将他们限制在陈家范围

    tf.name_scope('cgx_scope')语句重复执行几次，就会生成几个独立的命名空间，尽管表面上看起来都是“cgx_scope”，实际上
    tensorflow在每一次执行相同语句都会在后面加上“_序数”，加以区别
    
    此外，name_scope可以嵌套，类似于文件夹之中有文件夹的概念
    对tf.Variable 起作用
    tf.get_variable函数不受tf.name_scope的影响。
    
    算子也可用用name_scope 进行圈定
'''
if (x == 2):
    with tf.name_scope('cgx_scope'):
        a = tf.Variable(1, name='my_a')

    with tf.name_scope('cgx_scope'):
        b = tf.Variable(2, name='my_b')

    c = tf.add(a, b, name='my_add')

    print("a.name = " + a.name)
    print("b.name = " + b.name)

#   04 变量作用域的管理
'''
    tf.variable_scope()一般搭配tf.get_variable 进行使用
    tf.get_variable函数不受tf.name_scope的影响。
    作用包括：
    1、首先可以圈定变量名，类似于name_scope，同样可以嵌套
    2、变量重复定义时，采用reuse共享模式，就不会再次创建新变量
    3、要实现共享模式，首先决定了同name变量不能在非reuse模式下重复定义，其次同名variable_scope也不能重复出现
    实现共享模式：
    默认reuse=false
    tf.get_variable_scope().reuse_variables()
    tf.get_variable_scope().reuse = true
    with tf.variable_scope("foo", reuse=true)
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE)
    
    算子（ops）会受变量作用域（variable scope）影响，相当于隐式地打开了同名的名称作用域（name scope)
'''
# tf.variable_scope()用来指定变量的作用域，作为变量名的前缀，支持嵌套，针对tf.get_variable发挥作用，如下：
if (x == 2):
    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])  # foo/bar/v:0
    print(v)

    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        tf.get_variable_scope().reuse_variables()  # 如果不指定重用的话，就会新建
        # with tf.variable_scope('foo', reuse=True): 或者使用这一句
        v1 = tf.get_variable("v", [1])
    print(v)  # foo/v:0
    print(v1)  # foo/v:0

    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):  # 或者reuse=true
        v = tf.get_variable("v", [1])
        v1 = tf.get_variable("v", [1])
    print(v)
    print(v1)

    # 一个作用域可以作为另一个新的作用域的参数
    with tf.variable_scope("foo") as foo_scope:
        v = tf.get_variable("v", [1])
    with tf.variable_scope(foo_scope):
        w = tf.get_variable("w", [1])
    with tf.variable_scope(foo_scope, reuse=True):
        v1 = tf.get_variable("v", [1])
        w1 = tf.get_variable("w", [1])

    #不管作用域如何嵌套，当使用with tf.variable_scope()打开一个已经存在的作用域时，就会跳转到这个作用域
    with tf.variable_scope("foo") as foo_scope:
        assert foo_scope.name == "foo"  # assert（断言）Python assert（断言）用于判断一个表达式，
                                        # 在表达式条件为 false 的时候触发异常。
    with tf.variable_scope("bar"):
        with tf.variable_scope("baz") as other_scope:
            assert other_scope.name == "bar/baz"
            with tf.variable_scope(foo_scope) as foo_scope2:
                assert foo_scope2.name == "foo"  # Not changed.

    # https://www.cnblogs.com/MY0213/p/9208503.html#4385667 关于作用于的嵌套
    # TF.VARIABLE、TF.GET_VARIABLE、TF.VARIABLE_SCOPE以及TF.NAME_SCOPE关系
    # tf.get_variable函数不受tf.name_scope的影响。

    # variable scope的Initializers可以创递给子空间和tf.get_variable()函数，除非中间有函数改变，否则不变。
    '''
    tf.get_variable（）和tf.variable_scope函数中的 的initializer参数的选择如下
        tf.constant_initializer：常量初始化函数
        tf.random_normal_initializer：正态分布
        tf.truncated_normal_initializer：截取的正态分布
        tf.random_uniform_initializer：均匀分布
        tf.zeros_initializer：全部是0
        tf.ones_initializer：全是1
        tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
    '''
    with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
        v = tf.get_variable("v", [1])
        assert v.eval() == 0.4  # Default initializer as set above.
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3))
        assert w.eval() == 0.3  # Specific initializer overrides the default.
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
            assert v.eval() == 0.4  # Inherited default initializer.
        with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):
            v = tf.get_variable("v", [1])
            assert v.eval() == 0.2  # Changed default initializer.


