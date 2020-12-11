# BP ，更新阈值和权重，回归预测问题最后一层不带激活函数
# coding: UTF-8
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tkinter import _flatten

def load_data_wrapper(filename):
    lineData = []
    with open(filename) as txtData:
        lines = txtData.readlines()
        for line in lines:
            linedata = line.strip().split(',')
            lineData.append(linedata)
    return lineData

def splitData(dataset):
    Character= []
    Label = []
    for i in range(len(dataset)):
        Character.append([float(tk) for tk in dataset[i][1:-1]])
        Label.append(float(dataset[i][-1]))
    return Character, Label

def max_min_norm_x(dataset):
    min_data = []
    for i in range(len(dataset)):
        min_data.append(min(dataset[i]))
    new_min = min(min_data)
    max_data = []
    for i in range(len(dataset)):
        max_data.append(max(dataset[i]))
    new_max = max(max_data)
    data = np.array(dataset)
    data_x =[]
    for x in np.nditer(data, op_flags=['readwrite']):
        #x[...] = 2 * (x -new_min)/(new_max-new_min)-1
        x[...] = (x - new_min) / (new_max - new_min)
        #print('x[...]:',x[...])
        data_x.append(x[...])
    data_x3 = []
    for index in range(0, len(data_x), 3):
        data_x3.append([data_x[index], data_x[index+1], data_x[index+2]])
    #print("data_x3:",data_x3)
    return data_x3

def max_min_norm_y(dataset):
    new_min = min(dataset)
    new_max = max(dataset)
    data_y = []
    for i in range(len(dataset)):
        y = (dataset[i] -new_min)/(new_max-new_min)
        #y = 2 * (dataset[i] - new_min) / (new_max - new_min) - 1
        data_y.append(y)
        #print(y)
    return data_y

def de_max_min_norm_y(dataset1,dataset2):
    new_min = min(dataset1)
    new_max = max(dataset1)
    de_data_y = []
    for i in range(len(dataset2)):
        y = dataset2[i] * (new_max - new_min) + new_min
        de_data_y.append(y)
    return de_data_y
# 初始化参数
# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
def parameter_initialization(x, y, z):
    # 隐层阈值从（-5,5）之间的随机数
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)

    # 输出层阈值
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)

    # 输入层与隐层的连接权重
    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)

    # 隐层与输出层的连接权重
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)

    return weight1, weight2, value1, value2

#定义激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu(z):
    return np.where(z < 0, 0, z)

'''
weight1:输入层与隐层的连接权重
weight2:隐层与输出层的连接权重
value1:隐层阈值
value2:输出层阈值
'''
#训练过程
def train_process(dataset, labelset, weight1, weight2, value1, value2):
    # x为步长
    x = 0.05
    for i in range(len(dataset)):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.float64)
        print('inputset',inputset)
        print(inputset.shape)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)
        # 隐层输入
        input1 = np.dot(inputset, weight1).astype(np.float64)
        print("隐含层输入：", input1)
        # 隐层输出
        #output2 = relu(input1 - value1).astype(np.float64)

        output2 = sigmoid(input1 - value1).astype(np.float64)
        print("隐层输出:", output2)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)
        # 输出层输出
        output3 = input2 - value2
        #output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵运算表示 用的是平方误差
        #a = np.multiply(output3, 1 - output3) #最后一层激活函数求导
        #g = output3 - outputset #最后一层直接求导，无激活函数，为输出层阈值求导
        g = outputset - output3 #最后一层直接求导 ，为输出层阈值求导
        #g = np.multiply(a, outputset - output3)
        b = np.dot(g, np.transpose(weight2))
        #c = output2
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)  # 隐藏层之间阈值

        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        # 更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2

def test_process(dataset, labelset, weight1, weight2, value1, value2):
    pre_data = []
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        #output2 = relu(np.dot(inputset, weight1) - value1)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = np.dot(output2, weight2) - value2
        output3 = output3.tolist()
        pre_data.append(output3)
        pre_data = list(_flatten(pre_data))
        #pre_data = de_max_min_norm_y(labelset, pre_data)
        #output3 = sigmoid(np.dot(output2, weight2) - value2)
        #print("预测为%f, 实际为%f" % (output3, labelset[i]))
        # 返回预测值
    return pre_data

if __name__ == '__main__':
    iris_file = 'advertise.txt'
    Data = load_data_wrapper(iris_file)
    x, y = splitData(Data)
    print("x",x)
    x_norm = max_min_norm_x(x)
    y_norm = max_min_norm_y(y)
    #x为数据集的feature数据，y为label.
    x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size=0.3)
    # x_train = max_min_norm_x(x_train)
    # y_train = max_min_norm_y(y_train)
    # x_test = max_min_norm_x(x_test)
    # y_test = max_min_norm_y(y_test)
    weight1, weight2, value1, value2 = parameter_initialization(len(x_train[0]), 2, 1)
    #weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), len(dataset[0]), 1)
    for i in range(700):
        weight1, weight2, value1, value2 = train_process(x_train, y_train, weight1, weight2, value1, value2)
        #weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    print(weight1)
    print(weight2)
    print(value1)
    print(value2)
    pre = test_process(x_test, y_test, weight1, weight2, value1, value2)
    print("pre:",pre)
    print("y_test:",y_test)
    pre_org = np.array(pre) * (max(y) - min(y)) + min(y)
    y_test_org = np.array(y_test) * (max(y) - min(y)) + min(y)
    print("pre_org\n", pre_org)
    print("y_test_org\n", y_test_org)
    errors_std = np.std(np.array(pre) - np.array(y_test))
    errors_std_org = np.std(pre_org - y_test_org)
    print("errors_std:\n", errors_std)
    print("errors_std_org\n", errors_std_org)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.linspace(0, 60, 60)
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, y_test)
    plt.plot(x, pre, color='red', linewidth=1, linestyle='--')
    plt.xlim((0, 60))
    plt.ylim((0, 1))
    plt.xlabel('测试样本个数')
    plt.ylabel('归一化后预测的数值')
    plt.show()


