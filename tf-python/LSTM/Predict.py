
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义常量
BASE_DIR = os.path.abspath('.')
rnn_unit = 10
input_size = 7
output_size = 1
lr = 0.0006
weights = {
    'in': tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit,1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
    'out': tf.Variable(tf.constant(0.1,shape=[1,]))
}

# 读文件
f = open('dataset/dataset.csv') 
df = pd.read_csv(f)
# 读取属性open, close, low, high, volume, money, change, label
data = df.iloc[: , 2: 10].values

# 获取训练集数据
def get_train_data(batch_size = 60, time_step = 20, train_begin = 0, train_end = 5800): 
    batch_index = []
    data_train = data[train_begin: train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis = 0)) / np.std(data_train, axis = 0)
    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step): 
        if i % batch_size == 0: 
            batch_index.append(i)
        x = normalized_train_data[i: i+time_step, : 7]
        y = normalized_train_data[i: i+time_step, 7, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index, train_x, train_y

# 获取测试集数据
def get_test_data(time_step = 20, test_begin = 5800): 
    data_test = data[test_begin: ]
    mean = np.mean(data_test, axis = 0)
    std = np.std(data_test, axis = 0)
    normalized_test_data = (data_test - mean) / std
    size = (len(normalized_test_data) + time_step - 1) // time_step
    test_x, test_y = [], []  
    for i in range(size - 1): 
        x = normalized_test_data[i * time_step: (i + 1) * time_step, : 7]
        y = normalized_test_data[i * time_step: (i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step: , : 7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step: , 7]).tolist())
    return mean, std, test_x, test_y

# 构建神经网络变量
def lstm(X):      
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype = tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(
        cell, 
        input_rnn, 
        initial_state = init_state, 
        dtype = tf.float32
    )
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

# 预测并可视化
def prediction(time_step = 20): 
    X = tf.placeholder(tf.float32, shape = [None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess: 
        # 读取训练好的参数
        module_file  =  tf.train.latest_checkpoint(BASE_DIR)
        saver.restore(sess, module_file) 
        test_predict = []
        for i in range(len(test_x) - 1): 
            prob = sess.run(pred, feed_dict = {X:  [test_x[i]]})   
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y[: len(test_predict)]) / test_y[: len(test_predict)])
        # 可视化
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color = 'b')
        plt.plot(list(range(len(test_y))), test_y, color = 'r')
        plt.show()


prediction() 