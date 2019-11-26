#!/usr/bin/env python3
#设损失函数loss=（w+1）^2,令w初值为5，反向传播就是求最优w，即求最小loss对应的w值
%matplotlib inline
import tensorflow as tf
LEARNING_RATE_BASE = 0.1
#学习率衰减率
LEARNING_RATE_DECAY = 0.99
#喂多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE
LEARNING_RATE_STEP = 1

global_step = tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
#定义待优化参数w初值为5
w = tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播函数
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

#生成会话并训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 200
    for i in range(STEPS):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        if i%10 == 0:
            print('After %d steps: w is %f, learning_rate is %f, loss is %f.'%(i,w_val,learning_rate_val,loss_val))
            print(global_step_val)
print('Finish')

