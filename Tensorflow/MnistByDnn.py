'''
Created on 2017-11-21

@author: DongWenhao
'''

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from mnist import MNIST_train_data, MNIST_train_label, MNIST_test_data, MNIST_test_label

print('start')

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1):
    train_step.run(feed_dict={x:MNIST_train_data, y_: MNIST_train_label}) 

print('Train finish')

#print(y.eval(feed_dict={x: MNIST_test_data})) 


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (accuracy.eval(feed_dict={x: MNIST_test_data, y_: MNIST_test_label}))

 
