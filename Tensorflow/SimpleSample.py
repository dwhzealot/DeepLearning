import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

print('start')

l0 = np.array([[0,0,1],
           [0,1,1],
           [1,0,1],
           [1,1,1],
           [0,0,1]]
)

labels = np.array([[0],
              [1],
              [1],
              [0],
              [0]]
)


sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[5, 3])
y_ = tf.placeholder("float", shape=[5, 1])

W = tf.Variable(tf.zeros([3,1]))
b = tf.Variable(tf.zeros([3]))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    train_step.run(feed_dict={x:l0, y_: labels})


print('finish')
