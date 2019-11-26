#!/usr/bin/env python3
#导入模块，生成数据集
%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
SEED = 2

#基于seed产生随机数,准备虚拟的数据样本
rdm = np.random.RandomState(SEED)
X = rdm.randn(300,2)
#作为输入数据集的标签（正确答案）
Y_ = [int(x0*x0+x1*x1<2) for (x0,x1) in X]
#防止出错的写法Y_ = [[int(x0*x0+x1*x1<2)] for (x0,x1) in X]
Y_c = [['red' if y else 'blue'] for y in Y_]
#对数据集X和标签Y进行整理，第一个元素为-1表示随第二个参数计算得到，第二个元素表示列数
X = np.vstack(X).reshape(-1,2)
Y_ = np.vstack(Y_).reshape(-1,1)
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.show()

#定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2)+b2

#定义损失函数及反向传播方法，损失函数为MSE，反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y-y_))
#均方误差的损失函数加每一个正则化w的损失
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))


#定义反向传播方法，不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
#生产会话，训练STEPS
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 1000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%200 == 0:
            #不含正则化的损失函数
            loss_mse_val = sess.run(loss_mse,feed_dict={x:X,y_:Y_})
            print('After %d training steps,loss is: %f'%(i,loss_mse_val))
    #xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01生成二维网格坐标点
    xx,yy = np.mgrid[-3:3:.01, -3:3:.01]
    #将xx，yy拉直，并合并成一个2列的矩阵，得到一个网络坐标点的集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    #将网络坐标点喂入神经网络，probs为输出
    probs = sess.run(y,feed_dict={x:grid})
    #probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print('w1:\n',sess.run(w1))
    print('b1:\n',sess.run(b1))
    print('w2:\n',sess.run(w2))
    print('b2:\n',sess.run(b2))
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()

# 定义反向传播方法，包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
# 生产会话，训练STEPS
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 4000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 200 == 0:
            # 不含正则化的损失函数
            # loss_mse_val = sess.run(loss_mse,feed_dict={x:X,y_:Y_})
            # print('After %d training steps,loss is: %f'%(i,loss_mse_val))

            # 包含正则化的损失函数
            loss_total_val = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print('After %d training steps,loss is: %f' % (i, loss_total_val))
    # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx，yy拉直，并合并成一个2列的矩阵，得到一个网络坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网络坐标点喂入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x: grid})
    # probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print('w1:\n', sess.run(w1))
    print('b1:\n', sess.run(b1))
    print('w2:\n', sess.run(w2))
    print('b2:\n', sess.run(b2))
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
