#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import random
import numpy as np
from Activators import SigmoidActivator
import datetime as datetime

now = datetime.datetime.now()

# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, sample_num,
                 activator):
        '''
        #构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.sample_num = sample_num;
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
            (input_size, output_size))
        # 偏置项b
        #self.b = np.zeros((sample_num, 1))

        # 输出向量
        self.output = np.zeros((sample_num, output_size))
        
    def forward(self, input_array):
        '''
        #前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        w_size = self.W.shape
        #print('forward input_array len', len(input_array))  
        input_mat = np.ones((self.sample_num, w_size[0])) * input_array
        #print('forward input_mat', input_mat.shape)        
        #print('forward self.W size', self.W.shape)
        self.input = input_mat
        # 式2
        mul_mat = np.dot(input_mat, self.W)

        #print('forward mul_mat size', mul_mat.shape)
        #print('forward mul_mat ',mul_mat)
        #print('forward b size', self.b.shape)
        #mul_b = mul_mat + self.b
        #print('forward mul_b size', mul_b.shape)
        self.output = self.activator.forward(mul_mat)
        #print('forward out size', self.output.shape, '\n')
        #print('forward out ', self.output, '\n')
    def backward(self, delta_array):
        '''
        #反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        #print('backward, delta_array size', delta_array.shape)
        # 式8
        #print('backward, delta_array size', delta_array.shape)
        self.delta = self.activator.backward(self.input) * np.dot(
            delta_array,self.W.T)
        self.W_grad = np.dot(self.input.T,delta_array)
        #self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        #使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        #print('update, self.b size', self.b.shape)
        #print('update, self.b_grad size', self.b_grad.shape)
        #self.b += learning_rate * self.b_grad

    def dump(self):
        print ('W: %s' %self.W)


# 神经网络类
class Network(object):
    def __init__(self, layers, sample_num):
        '''
        #构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1], sample_num,
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        #使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        #print('predict, out_put size',output.shape)
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        #训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], 
                    data_set[d], rate)
                if d % 500 == 0:
                    print(now, 'train, finish train 500 sample')

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        #检查梯度
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print ('weights(%d,%d): expected - actural %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i,j]))



