# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
# 一堆点用线的方式去学习


"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def add_layer(inputs, in_size, out_size, activation_function=None):
    # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b  #线性函数，如果是非线性方程哪？
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# # Make up some real data，＃编写一些真实的数据
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise

#导入自己的数据测试
f=open('aapl.csv')
df=pd.read_csv(f)  #读入股票数据,df=pd.read_csv(f)
x_data=np.array(df['High'])   #获取最高价序列
y_data=np.array(df['Low'])
#以折线图展示data
# plt.figure()
# plt.plot(x_data,'o')
# plt.plot(y_data,'o')
# plt.show()
print(x_data,y_data)


# define placeholder for inputs to network，定义输入到网络的占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data，预测与真实数据之间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# important step
sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
fig = plt.figure(1)
ax = fig.add_subplot(2,1,2)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

#自己加的
fig = plt.subplot(211)
plt.plot(x_data,y_data)
plt.show()




for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(1)
