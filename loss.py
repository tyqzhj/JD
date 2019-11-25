#!/usr/bin/env python3
#倒入模块，生成数据集
%matplotlib inline
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]
#定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)
#定义损失函数及反向传播方法，损失函数为MSE，反向传播方法为梯度下降
#loss_mse = tf.reduce_mean(tf.square(y_-y))
#自定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
#交叉熵损失函数
#最后的结果感觉不正确，后续待解决
#ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
#cem = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#生产会话，训练STEPS
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = (i*BATCH_SIZE)%32 + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%500 == 0:
            print('After %d training steps,loss on all data is:'%(i),sess.run(loss,feed_dict={x:X,y_:Y_}))
    print('w1 is:\n',sess.run(w1))

