# coding=gbk  
import numpy as np  
import struct  
  
def loadImageSet(filename, img_num):  
    # ��ȡ�������ļ� 
    binfile = open(filename, 'rb')  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0) # ȡǰ4������������һ��Ԫ��  
  
    offset = struct.calcsize('>IIII')  # ��λ��data��ʼ��λ��  
    imgNum = img_num  #head[1]
    width = head[2]  
    height = head[3]  
  
    bits = imgNum * width * height  # dataһ����60000*28*28������ֵ  
    bitsString = '>' + str(bits) + 'B'  # fmt��ʽ��'>47040000B'  
  
    imgs = struct.unpack_from(bitsString, buffers, offset) # ȡdata���ݣ�����һ��Ԫ��  
  
    binfile.close()  
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshapeΪ[60000,784]������  
  
    return imgs,head
  
  
def loadLabelSet(filename, label_num):  
  
    binfile = open(filename, 'rb') # ���������ļ�  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0) # ȡlabel�ļ�ǰ2��������  
  
    labelNum = label_num #head[1]  
    offset = struct.calcsize('>II')  # ��λ��label���ݿ�ʼ��λ��  
  
    numString = '>' + str(labelNum) + "B" # fmt��ʽ��'>60000B'  
    labels = struct.unpack_from(numString, buffers, offset) # ȡlabel����  
  
    binfile.close()  
    label_vec = []
    for i in range(labelNum):
        for j in range(10):
            if j == labels[i]:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)

    labels = np.reshape(label_vec, [labelNum,10]) # ת��Ϊ�б�(һά����)  
  
    return labels,head
'''
    def norm(self, label):
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
'''
file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'
MNIST_train_data, MNIST_train_data_head = loadImageSet(file1, 6000)
MNIST_train_label, MNIST_train_label_head = loadLabelSet(file2, 6000)
MNIST_test_data, MNIST_test_data_head = loadImageSet(file3, 500)
MNIST_test_label, MNIST_test_label_head = loadLabelSet(file4, 500)

#print(MNIST_test_label)