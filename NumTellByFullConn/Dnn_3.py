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
m = 500
m_test = 100
layer_num = 3
epoch = 2000
W_ini_coe = 0.1
learn_rate = 0.1

TrainSet = getDataSet(file1, file2)

print('Dnn3 start')
activator = SigmoidActivator()
n = [5,3,10]

network = FullConnNetwork(m, s, layer_num, W_ini_coe ,activator, learn_rate, n)

print('Training start')
for i in range(epoch):
    X, Y = TrainSet.random_block(m)
    network.Train(X, Y)

print('Training end\nTesting start')
TestSet = getDataSet(file3, file4)
X_test,Lable_test = TestSet.random_block(m_test)
Test_result = network.ForwardPropagation(X_test)

print(evaluate(Test_result, Lable_test))
print('testing end')

print('Dnn3 end')
