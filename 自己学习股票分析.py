# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf12_plot_result/full_code.py

# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # Weights = tf.Variable(tf.random_normal(10.0,100.0,)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise


# plt.figure()
# plt.plot(x_data,y_data,'o')
# plt.show()



# 导入自己的数据测试
f = open('aapl.csv')
df = pd.read_csv(f)  #读入股票数据,df=pd.read_csv(f)
x_data = np.array(df['Low'])[:, np.newaxis]  #获取最高价序列
y_data = np.array(df['High'])[:, np.newaxis]

# x_data= x_data/np.max(x_data)
x_data= x_data * 0.01
# y_data= y_data/np.max(y_data)
y_data= y_data * 0.01


# plt.figure()
# plt.scatter(x_data,y_data)
# plt.plot(x_data,y_data)
# plt.show()
# print(x_data)
# print(y_data)



# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)


# the error between prediction and real data，预测与真实数据之间的误差

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

# f.train.GradientDescentOptimizer()中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# important step
sess = tf.Session()
# sess.run(tf.global_variables_initializer())


# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
fig = plt.figure(1)
ax = fig.add_subplot(1,2,1)
ax.scatter(x_data, y_data)

plt.ion()


# 机器开始学习
# 每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    if i % 500 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=2)
        # plt.text(0.5, 0.5, 'Loss=%.4f' % i, fontdict={'size': 20, 'color': 'red'})
        plt.pause(1)


plt.ioff()



# 导入自己的数据测试
f = open('aapl.csv')
df = pd.read_csv(f)  #读入股票数据,df=pd.read_csv(f)
x_data = np.array(df['Low'])[:, np.newaxis]  #获取最高价序列
y_data = np.array(df['L'])[:, np.newaxis]

# x_data= x_data/np.max(x_data)
x_data= x_data * 0.01
# y_data= y_data/np.max(y_data)
y_data= y_data * 0.01


# plt.figure()
# plt.scatter(x_data,y_data)
# plt.plot(x_data,y_data)
# plt.show()
# print(x_data)
# print(y_data)



# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)


# the error between prediction and real data，预测与真实数据之间的误差

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

# f.train.GradientDescentOptimizer()中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# important step
sess = tf.Session()
# sess.run(tf.global_variables_initializer())


# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
fig = plt.figure(1)
ax = fig.add_subplot(1,2,2)
ax.scatter(x_data, y_data)

plt.ion()


# 机器开始学习
# 每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    if i % 500 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=2)
        # plt.text(0.5, 0.5, 'Loss=%.4f' % i, fontdict={'size': 20, 'color': 'red'})
        plt.pause(1)


plt.ioff()
plt.show()