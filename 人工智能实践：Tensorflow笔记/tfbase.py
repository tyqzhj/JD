#!/usr/bin/env python3
%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 8
#控制每次数据一样，利于debug
seed = 23455
#基于seed产生随机数,【1】准备虚拟的数据样本
rng = np.random.RandomState(seed)
X = rng.rand(32,2)
#数据标注
Y = [[int(x0+x1<1)] for (x0,x1) in X]
print('X:\n',X)
print('Y:\n',Y)
#定义神经网络的输入、参数和输出
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random.normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random.normal([3,1],stddev=1,seed=1))
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#定义损失函数及【3】反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# 【4】生成会话,训练STEPS
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前未经训练的参数取值
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))

    # 训练模型
    STEPS = 6000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X, y_: Y})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After %d training steps, loss on all data is %g' % (i, total_loss))
    # 输出训练后的参数取值
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))