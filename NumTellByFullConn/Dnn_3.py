# -*- coding: UTF-8 -*-
'''
Created on 2017��11��28��

@author: DongWenhao
'''
from DnnClass import *
from Activators import *
from mnist import *

file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'


s = 784
m = 5000
m_test = 100
layer_num = 3
epoch = 1000
W_ini_coe = 0.1
learn_rate = 0.1

X,Y = mnist_load_random_block(file1, file2, m)

print('Dnn3 start')
activator = SigmoidActivator()
n = [5,3,10]

network = FullConnNetwork(m, s, layer_num, W_ini_coe ,activator, learn_rate, n)

print('Training start')
for i in range(epoch):
    network.Train(X, Y)

print('Training end\nTesting start')
X_test,Lable_test = mnist_load_random_block(file3, file4, m_test)
Test_result = network.ForwardPropagation(X_test)
#print('Test_result',Test_result)

print(evaluate(Test_result, Lable_test))
print('testing end')

print('Dnn3 end')
