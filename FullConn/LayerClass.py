# -*- coding: UTF-8 -*-
'''
Created on 2017-11-27

@author: DongWenhao
'''
from mnist import *
from Activators import *
import numpy as np

class HideLayer(object):
    def __init__(self, m, n, column, W_ini_coe ,activator, learn_rate):
        '''
        #构造函数
        m: 样本个数
        n:本层神经元个数，即行数
        colum: W的列数 
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        OutputLayer_flag: 是否是输出层
        '''
        self.m = m
        self.n = n
        self.column = column
        self.activator = activator
        self.W = np.random.randn(n,column) * W_ini_coe
        self.W_before_update = np.zeros(n,column)
        self.b = np.zeros((n, 1))
        self.learn_rate = learn_rate
        
    def forward(self, input_mat):
        '''
        #前向计算
        input_mat: 本层的输入, 即为上一层的输出 A
        '''
        self.input = input_mat
        self.Z = np.dot(self.W, input_mat) + self.b
        self.A = self.activator.forward(self.Z)
        self.W_before_update = self.W
        
    def backward(self, dZ_next_layer, W_next_layer):
        '''
        #反向计算W和b的梯度
        dZ_next_layer: 下一层反向传播过来的 dZ
        W_next_layer: 下一层的权重W
        '''
        self.dZ = (np.dot(W_next_layer.T,dZ_next_layer)) * (self.activator.backward(self.Z))
        dW = (np.dot(self.dZ, self.input.T))/self.m
        db = (np.sum(self.dZ, axis=1, keepdims=True))/self.m

        self.W -= self.learn_rate * dW
        self.b -= self.learn_rate * db
        return self.dZ, self.W_before_update
        
class OutputLayer(object):
    def __init__(self, m, n, column, W_ini_coe ,activator, learn_rate):
        '''
        #构造函数
        m: 样本个数
        n:本层神经元个数，即行数
        colum: W的列数 
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        '''
        self.m = m
        self.n = n
        self.column = column
        self.activator = activator
        self.W = np.random.randn(n,column) * W_ini_coe
        self.W_before_update = np.zeros(n,column)
        self.b = np.zeros((n, 1))
        self.learn_rate = learn_rate
        
    def forward(self, input_mat):
        '''
        #前向计算
        input_mat: 本层的输入, 即为上一层的输出 A
        '''
        self.input = input_mat
        self.Z = np.dot(self.W, input_mat) + self.b
        self.A = self.activator.forward(self.Z)
        self.W_before_update = self.W
        
    def backward(self, Y):
        '''
        #反向计算W和b的梯度
        Y : label
        '''
        
        self.dZ = self.A - Y
        dW = (np.dot(self.dZ, self.input.T))/self.m
        db = (np.sum(self.dZ, axis=1, keepdims=True))/self.m

        self.W -= self.learn_rate * dW
        self.b -= self.learn_rate * db
        return self.dZ, self.W_before_update
