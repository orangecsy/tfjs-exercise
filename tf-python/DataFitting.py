'''
目的：用平面拟合数据点
算法：随机生成100个点，构建线性模型及优化器，训练200次
输入：无
输出：每十次训练输出“当前步 权重W 一次项b”
'''

import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据100个
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
W = tf.Variable(tf.random_uniform([1, 2]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
# 已弃用 init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(W), sess.run(b))