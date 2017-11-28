# -*- coding: UTF-8 -*-
'''
Created on 2017年11月26日

@author: DongWenhao
'''
from mnist import *
from Activators import *
import numpy as np

def get_result(vec):
    max_value_index = 0
    max_value = 0
    vec_size = vec.shape
    m = vec_size[0]
    result = np.zeros(m)
    for i in range(m):
        max_value = 0
        for j in range(10):
            if vec[i][j] > max_value:
                max_value = vec[i][j]
                result[i] = j
    return result
    

def evaluate(test_result, test_labels):
    error = 0
    total = len(test_result)
    label = get_result(test_labels)
    predict = get_result(test_result)
    for i in range(total):
        if label[i] != predict[i]:
            error += 1
    return float(error) / float(total)

file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'
activate = SigmoidActivator()

print('Dnn_2 training start\n')
train_num = 100

m = train_num
W_elm = 4
W1 = np.random.randn(W_elm,784) * 0.1
b1 = np.zeros((W_elm,1))
W2 = np.random.randn(10,W_elm) * 0.1
b2 = np.zeros((10,1))

for i in range(5000):

    X,Y = mnist_load_random_block(file1, file2, m)

    Z1 = np.dot(W1,X) + b1
    A1 = activate.forward(Z1)
    Z2 = np.dot(W2,Z1) + b2
    A2 = activate.forward(Z2)


    dZ2 = A2 - Y
    dW2 = (np.dot(dZ2, A1.T))/m
    db2 = (np.sum(dZ2, axis=1, keepdims=True))/m
    
    dZ1 = (np.dot(W2.T,dZ2)) * (activate.backward(Z1))
    dW1 = (np.dot(dZ1, X.T))/m
    db1 = (np.sum(dZ1, axis=1, keepdims=True))/m
    
    W1 -= 0.1 * dW1
    b1 -= 0.1 * db1
    W2 -= 0.1 * dW2
    b2 -= 0.1 * db2

print('\nDnn_2 training end')
print('\nDnn_2 testing start')

test_num = 100
X_test,Y_test = mnist_load_random_block(file3,file4, test_num)

Z1_test = np.dot(W1,X_test) + b1
A1_test = activate.forward(Z1_test)
Z2_test = np.dot(W2,A1_test) + b2
A2_test = activate.forward(Z2_test)

print('ratio :',evaluate(A2_T, Y_T))

print('\nDnn_2 testing end')

