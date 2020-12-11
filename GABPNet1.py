import numpy as np
from sklearn.model_selection import train_test_split
import random
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import heapq
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



# 求染色体长度
def getEncodeLength(decisionvariables, delta):
     # 将每个变量的编码长度放入数组
     lengths = []
     uper0 = decisionvariables[0][1]
     low0 = decisionvariables[0][0]
     res = fsolve(lambda x: ((uper0 - low0) / delta - 2 ** x + 1), 6)
     length0 = int(np.ceil(res[0]))
     uper1 = decisionvariables[0][1]
     low1 = decisionvariables[0][0]
     res = fsolve(lambda x: ((uper1 - low1) / delta - 2 ** x + 1), 2)
     length1 = int(np.ceil(res[0]))
     uper2 = decisionvariables[0][1]
     low2 = decisionvariables[0][0]
     res = fsolve(lambda x: ((uper2 - low2) / delta - 2 ** x + 1), 2)
     length2 = int(np.ceil(res[0]))
     uper3 = decisionvariables[0][1]
     low3= decisionvariables[0][0]
     res = fsolve(lambda x: ((uper3 - low3) / delta - 2 ** x + 1), 1)
     length3= int(np.ceil(res[0]))
     lengths.append(length0)
     lengths.append(length1)
     lengths.append(length2)
     lengths.append(length3)
     print("染色体长度:", lengths)
     # for decisionvar in decisionvariables:
     #     uper = decisionvar[1]
     #     low = decisionvar[0]
     #     # res()返回一个数组
     #     res = fsolve(lambda x: ((uper - low) / delta - 2 ** x + 1), 10)
     #     # ceil()向上取整
     #     length = int(np.ceil(res[0]))
     #     lengths.append(length)
     #print("染色体长度:", lengths)
     return lengths,length0,length1,length2,length3
# 随机生成初始化种群
def getinitialPopulation(length,length0,length1,length2,length3, populationSize):
    chromsomes = np.zeros((populationSize, length), dtype=np.int)
    #print("len",length)  # 每个变量的值相加 11
    chromsomes0 = np.zeros((populationSize, length0), dtype=np.int)
    chromsomes1 = np.zeros((populationSize, length1), dtype=np.int)
    chromsomes2 = np.zeros((populationSize, length2), dtype=np.int)
    chromsomes3 = np.zeros((populationSize, length3), dtype=np.int)
    for popusize in range(populationSize):
        # np.random.randit()产生[0,2)之间的随机整数，第三个参数表示随机数的数量
        chromsomes[popusize, :] = np.random.randint(0, 2, length)
        chromsomes0[popusize, :] = chromsomes[popusize, :][0:6]
        chromsomes1[popusize, :] = chromsomes[popusize, :][6:8]
        chromsomes2[popusize, :] = chromsomes[popusize, :][8:10]
        chromsomes3[popusize, :] =chromsomes[popusize, :][10:11]
        #print('chromsomes[popusize, :]',chromsomes[popusize, :])  #100条，一个种群，100个体
    # print(chromsomes)
    # print(chromsomes0)
    # print(chromsomes1)
    # print(chromsomes2)
    # print(chromsomes3)
    # print(chromsomes.shape) #(100,11)
    return chromsomes,chromsomes0,chromsomes1,chromsomes2,chromsomes3

def getPopulation(population, populationSize):
    population0 = np.zeros((populationSize, 6), dtype=np.int)
    population1 = np.zeros((populationSize, 2), dtype=np.int)
    population2 = np.zeros((populationSize, 2), dtype=np.int)
    population3 = np.zeros((populationSize, 1), dtype=np.int)
    for popusize in range(populationSize):
        # np.random.randit()产生[0,2)之间的随机整数，第三个参数表示随机数的数量
        population0[popusize, :] = population[popusize, :][0:6]
        population1[popusize, :] = population[popusize, :][6:8]
        population2[popusize, :] = population[popusize, :][8:10]
        population3[popusize, :] = population[popusize, :][10:11]
    return population0, population1, population2, population3
# 随机生成初始化种群
# def getinitialPopulation(length, populationSize):
#     chromsomes = np.zeros((populationSize, length), dtype=np.int)
#     print("len",length)  # 每个变量的值相加 17*4 = 68
#     for popusize in range(populationSize):
#         # np.random.randit()产生[0,2)之间的随机整数，第三个参数表示随机数的数量
#         chromsomes[popusize, :] = np.random.randint(0, 2, length)
#         #print('chromsomes[popusize, :]',chromsomes[popusize, :])  #100条，一个种群，100个体
#     print(chromsomes)
#     print(chromsomes.shape) #(100,33) (100,68)
#     return chromsomes

 # 染色体解码得到表现形的解
def getDecode(population,population0,population1,population2,population3, encodelength, decisionvariables):
    population_decimal = (
                (population.dot(np.power(2, np.arange(sum(encodelength))[::-1])) / np.power(2, len(encodelength)) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] +decisionvariables[0][1]))
    for i in range(population0.shape[1]):
        population_w1 = (
                (population0.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] +decisionvariables[0][1]))
    #print("population_w1", population_w1)
    for i in range(population1.shape[1]):
        population_v1 = (
                (population1.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] + decisionvariables[0][1]))
    #print("population_v1", population_v1)
    for i in range(population2.shape[1]):
        population_w2 = (
                (population2.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] + decisionvariables[0][1]))
    #print("population_w2", population_w2)
    for i in range(population3.shape[1]):
        population_v2 = (
                (population3.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * ( decisionvariables[0][0] + decisionvariables[0][1]))
    return population_decimal,population_w1,population_v1,population_w2,population_v2  #(100,2) 100个种群中的两个变量转化成10进制

def getDecode1(p0,p1,p2,p3, decisionvariables):
    for i in range(len(p0)):
        population_w1 = (
                (p0.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] +decisionvariables[0][1]))
    #print("population_w1", population_w1)
    for i in range(len(p1)):
        population_v1 = (
                (p1.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] + decisionvariables[0][1]))
    #print("population_v1", population_v1)
    for i in range(len(p2)):
        population_w2 = (
                (p2.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * (decisionvariables[0][0] + decisionvariables[0][1]))
    #print("population_w2", population_w2)
    for i in range(len(p3)):
        population_v2 = (
                (p3.dot(np.power(2, 0)) / np.power(2, 1) - 0.5) *
                (decisionvariables[0][1] - decisionvariables[0][0]) + 0.5 * ( decisionvariables[0][0] + decisionvariables[0][1]))
    return population_w1,population_v1,population_w2,population_v2
 # 染色体解码得到表现形的解
# def getDecode(population, encodelength, decisionvariables, delta):
#      # 得到population中有几个元素
#      populationsize = population.shape[0]  # 100&200
#      #print('populationsize',populationsize)
#      length = len(encodelength)  # 2
#      #print('length',length)
#      decodeVariables = np.zeros((populationsize, length), dtype=np.float) #(100,2)
#      #print('decodeVariables',decodeVariables)
#      # 将染色体拆分添加到解码数组decodeVariables中
#      for i, populationchild in enumerate(population): #population 是(100,33)数组
#          # 设置起始点
#          start = 0
#          for j, lengthchild in enumerate(encodelength): # encodelength = [18,15]
#              power = lengthchild - 1
#              decimal = 0
#              for k in range(start, start + lengthchild):
#                  # 二进制转为十进制
#                  decimal += populationchild[k] * (2 ** power)
#                  power = power - 1
#              # 从下一个染色体开始
#              start = lengthchild
#              lower = decisionvariables[j][0]
#              uper = decisionvariables[j][1]
#              # 转换为表现形
#              decodevalue = lower + decimal * (uper - lower) / (2 ** lengthchild - 1)
#              # 将解添加到数组中
#              decodeVariables[i][j] = decodevalue
#      print("decodeVariables",decodeVariables)
#      return decodeVariables  #(100,2) 100个种群中的两个变量转化成10进制


 # 得到每个个体的适应度值及累计概率
def getFitnessValue(func, decode,w1,v1,w2,v2):

    # 得到种群的规模和决策变量的个数
    #print("decode.shape",decode.shape)
    popusize = decode.shape[0]  #100
    # 初始化适应度值空间
    fitnessValue = np.zeros((popusize, 1)) #(100,1)
    #print("fitnessValue.shape",fitnessValue.shape)
    for popunum in range(popusize):
        fitnessValue[popunum] = func(x_train,y_train,w1,v1,w2,v2,3,2,1,popusize)[-1]
     # 得到每个个体被选择的概率
    probability = fitnessValue / np.sum(fitnessValue)
    # 得到每个染色体被选中的累积概率，用于轮盘赌算子使用
    cum_probability = np.cumsum(probability)
    return fitnessValue, cum_probability


 # 选择新的种群
def selectNewPopulation(decodepopu, cum_probability):
    # 获取种群的规模和
    m, n = decodepopu.shape  #100,33
    # 初始化新种群
    newPopulation = np.zeros((m, n))
    for i in range(m):
         # 产生一个0到1之间的随机数
        randomnum = np.random.random()
        # 轮盘赌选择
        for j in range(m):
            if (randomnum < cum_probability[j]):
                newPopulation[i] = decodepopu[j]
                break
    #print('newPopulation',newPopulation)
    return newPopulation


 # 新种群交叉
def crossNewPopulation(newpopu, prob):
    m, n = newpopu.shape
    # uint8将数值转换为无符号整型
    numbers = np.uint8(m * prob)
     # 如果选择的交叉数量为奇数，则数量加1
    if numbers % 2 != 0:
        numbers = numbers + 1     # 初始化新的交叉种群
    updatepopulation = np.zeros((m, n), dtype=np.uint8)     # 随机生成需要交叉的染色体的索引号
    index = random.sample(range(m), numbers)     # 不需要交叉的染色体直接复制到新的种群中
    for i in range(m):
        if not index.__contains__(i):
            updatepopulation[i] = newpopu[i]
     # 交叉操作
    j = 0
    while j < numbers:
         # 随机生成一个交叉点，np.random.randint()返回的是一个列表
        crosspoint = np.random.randint(0, n, 1)
        crossPoint = crosspoint[0]
        # a = index[j]
        # b = index[j+1]
        updatepopulation[index[j]][0:crossPoint] = newpopu[index[j]][0:crossPoint]
        updatepopulation[index[j]][crossPoint:] = newpopu[index[j + 1]][crossPoint:]
        updatepopulation[index[j + 1]][0:crossPoint] = newpopu[j + 1][0:crossPoint]
        updatepopulation[index[j + 1]][crossPoint:] = newpopu[index[j]][crossPoint:]
        j = j + 2
    return updatepopulation


 # 变异操作
def mutation(crosspopulation, mutaprob):
     # 初始化变异种群
     mutationpopu = np.copy(crosspopulation)
     m, n = crosspopulation.shape
     # 计算需要变异的基因数量
     mutationnums = np.uint8(m * n * mutaprob)
     # 随机生成变异基因的位置
     mutationindex = random.sample(range(m * n), mutationnums)
     # 变异操作
     for geneindex in mutationindex:
         # np.floor()向下取整返回的是float型
         row = np.uint8(np.floor(geneindex / n))
         colume = geneindex % n
         if mutationpopu[row][colume] == 0:
             mutationpopu[row][colume] = 1
         else:
             mutationpopu[row][colume] = 0
     return mutationpopu

 # 找到重新生成的种群中适应度值最大的染色体生成新种群
def findMinPopulation(population, minevaluation, minSize):
     #将数组转换为列表
     minevalue = minevaluation.flatten()
     #maxevaluelist = maxevalue.tolist()
     minevaluelist = minevalue.tolist()
     # 找到前10个适应度最小的染色体的索引
     #maxIndex = map(maxevaluelist.index, heapq.nlargest(10, maxevaluelist))
     minIndex = map(minevaluelist.index, heapq.nsmallest(10, minevaluelist))

     index = list(minIndex)
     print("index",index)
     colume = population.shape[1] #11
     #print("colume",colume)
     # 根据索引生成新的种群
     minPopulation = np.zeros((minSize, colume))
     i = 0
     for ind in index:
         minPopulation[i] = population[ind]
         i = i + 1
     return np.uint8(minPopulation)

 # 适应度函数，使用lambda可以不用在函数总传递参数
#def fitnessFunction():
     #return lambda a, b: 21.5 + a * np.sin(4 * np.pi * a) + b * np.sin(20 * np.pi * b)
def fitnessFunction(dataset, labelset, temp1,temp2,temp3,temp4,inputnum , hiddennum, outputnum,num):
    # 最起码得是个（300,3,2）等的矩阵
    # x为步长
    x = 0.05
    # print("temp1",temp1)
    # print(temp1.shape)
    # print("temp2", temp2)
    # print(temp2.shape)
    # print("temp3", temp3)
    # print(temp3.shape)
    # print("temp4", temp4)
    # print(temp4.shape)
    if num !=0:
        Value1 = temp2.reshape(num, 1, hiddennum)
        # print(" Value1", Value1.shape) #(100,1,2)
        Value2 = temp4.reshape(num, outputnum, outputnum)
        # print(" Value2",Value2.shape) #(100,1,1)
        Weight1 = temp1.reshape(num, inputnum, hiddennum)
        # print(" Weight1", Weight1.shape) #(100,3,2)
        Weight2 = temp3.reshape(num, hiddennum, outputnum)
        # print(" Weight2", Weight2.shape) #(100,2,1)
        errors_net = []
        for v1, v2, w1, w2 in zip(Value1, Value2, Weight1, Weight2):
            # print("w1", w1)
            # print("v1", v1)
            # print("v2", v2)
            # print("w2", w2)
            errors = []
            for i in range(len(dataset)):
                # 输入数据
                inputset = np.mat(dataset[i]).astype(np.float64)
                # print("inputset", i, inputset)
                # print(inputset.shape)  # (1,3)
                # 数据标签
                outputset = np.mat(labelset[i]).astype(np.float64)
                # print('outputset', outputset)
                # print(outputset.shape)  # (1,1)
                # 隐层输入
                # input1 = np.einsum("ijk,ikn->ijn", inputset, Weight1).astype(np.float64)
                # input1 = np.dot(inputset, Weight1).astype(np.float64)
                input1 = np.dot(inputset, w1).astype(np.float64)
                # print("隐含层输入：", input1)
                # print(input1.shape)  # (1,2)
                # 隐层输出
                # output2 = relu(input1 - value1).astype(np.float64)

                # output2 = sigmoid(input1 - Value1).astype(np.float64)
                output2 = sigmoid(input1 - v1).astype(np.float64)
                # print("隐层输出:", output2)  # (1,2)
                # 输出层输入
                # input2 = np.einsum("ijk,ikn->ijn", output2, Weight2).astype(np.float64)
                # input2 = np.dot(output2, Weight2).astype(np.float64)
                input2 = np.dot(output2, w2).astype(np.float64)
                # 输出层输出
                # output3 = input2 - Value2
                output3 = input2 - v2
                # print("输出层输出", output3)
                # print(output3.shape)  # (1,1)
                # output3 = sigmoid(input2 - value2).astype(np.float64)
                error = abs(output3 - y_train[i])
                errors.append(error)

                # 更新公式由矩阵运算表示 用的是平方误差
                # a = np.multiply(output3, 1 - output3) #最后一层激活函数求导
                # g = output3 - outputset #最后一层直接求导，无激活函数，为输出层阈值求导
                g = outputset - output3  # 最后一层直接求导 ，为输出层阈值求导
                # print("g", g)
                # print(g.shape)  # (1,1)
                # g = np.multiply(a, outputset - output3)
                # b = np.dot(g, np.transpose(Weight2))
                b = np.dot(g, np.transpose(w2))

                # b = np.einsum("ijk,ikn->ijn", g, Weight2.transpose(0, 2, 1))
                # print("b", b.shape)  # (1,2)
                # c = output2
                c = np.multiply(output2, 1 - output2)
                # print("c", c)
                # print(c.shape)  # (1,2)
                e = np.multiply(b, c)  # 隐藏层之间阈值

                value1_change = -x * e
                value2_change = -x * g
                # print("value2_change:", value2_change)
                # print(value2_change.shape)  # (1,1)
                # weight1_change = x * np.einsum("ijk,ikn->ijn", inputset.transpose(0, 2, 1), e)
                # weight2_change = x * np.einsum("ijk,ikn->ijn", output2.transpose(0, 2, 1), g)
                weight1_change = x * np.dot(np.transpose(inputset), e)
                weight2_change = x * np.dot(np.transpose(output2), g)
                # print("weight1_change", weight1_change)
                # print(weight1_change.shape)
                # print('dataset[i]_temp',dataset[i])
                # 更新参数
                # print("Value1", Value1)
                # Value1 += value1_change
                # Value2 += value2_change
                # Weight1 += weight1_change
                # Weight2 += weight2_change
                v1 += value1_change
                v2 += value2_change
                w1 += weight1_change
                w2 += weight2_change
            # print("errors",errors)
            error_net = np.array(min(errors)).flatten()
            errors_net.append(error_net)  # (100,1)
            # print("error_net", error_net)
            # print("errors_net",errors_net)
            # errors_net1 = np.array(errors_net)
            # print("errors_net1",errors_net1.shape)
        return w1, w2, v1, v2, output3, error_net
    else:
        Value1 = temp2.reshape(1, hiddennum)
        #print(" Value1", Value1.shape) #(1,2)
        Value2 = temp4.reshape(outputnum, outputnum)
        # print(" Value2",Value2.shape) #(1,1)
        Weight1 = temp1.reshape(inputnum, hiddennum)
        #print(" Weight1", Weight1.shape) #(3,2)
        Weight2 = temp3.reshape(hiddennum, outputnum)
        # print(" Weight2", Weight2.shape) #(2,1)
        for i in range(len(dataset)):
            # 输入数据
            inputset = np.mat(dataset[i]).astype(np.float64)
            # print('inputset', inputset)
            # print(inputset.shape)
            # 数据标签
            outputset = np.mat(labelset[i]).astype(np.float64)
            # 隐层输入
            input1 = np.dot(inputset, Weight1).astype(np.float64)
            #print("隐含层输入：", input1)
            # 隐层输出
            # output2 = relu(input1 - value1).astype(np.float64)

            output2 = sigmoid(input1 - Value1).astype(np.float64)
            #print("隐层输出:", output2)
            # 输出层输入
            input2 = np.dot(output2, Weight2).astype(np.float64)
            # 输出层输出
            output3 = input2 - Value2
            # output3 = sigmoid(input2 - value2).astype(np.float64)

            # 更新公式由矩阵运算表示 用的是平方误差
            # a = np.multiply(output3, 1 - output3) #最后一层激活函数求导
            # g = output3 - outputset #最后一层直接求导，无激活函数，为输出层阈值求导
            g = outputset - output3  # 最后一层直接求导 ，为输出层阈值求导
            # g = np.multiply(a, outputset - output3)
            b = np.dot(g, np.transpose(Weight2))
            # c = output2
            c = np.multiply(output2, 1 - output2)
            e = np.multiply(b, c)  # 隐藏层之间阈值

            value1_change = -x * e
            value2_change = -x * g
            weight1_change = x * np.dot(np.transpose(inputset), e)
            weight2_change = x * np.dot(np.transpose(output2), g)
            #print("Value1aaa",Value1)
            # 更新参数
            Value1 += value1_change
            Value2 += value2_change
            Weight1 += weight1_change
            Weight2 += weight2_change
    return Weight1,Weight2,Value1,Value2,
def parameter_initialization(opt):
    # 输入层与隐层的连接权重
    weight1 =opt[0:6]
    # 隐层与输出层的连接权重
    weight2 =opt[6:8]
    # 隐层阈值
    value1 = opt[8:10]
    # 输出层阈值
    value2 = opt[10:11]

    return weight1, weight2, value1, value2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))



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
if __name__ == "__main__":
    iris_file = 'advertise.txt'
    Data = load_data_wrapper(iris_file)
    x, y = splitData(Data)
    print("x", x)
    x_norm = max_min_norm_x(x)
    y_norm = max_min_norm_y(y)
    # x为数据集的feature数据，y为label.
    x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, test_size=0.3)
    optimalvalue = []
    optimalvariables = []

    # 两个决策变量的上下界，多维数组之间必须加逗号
    # decisionVariables = [[-3.0, 12.1], [4.1, 5.8]]
    decisionVariables = [[-5.0, 5.0]] * 4  # 参数： W1, W2, V1, V2
    # 精度
    delta = 0.0001
    # 获取染色体长度
    EncodeLength, L0, L1, L2, L3 = getEncodeLength(decisionVariables, delta)
    # 种群数量
    initialPopuSize = 10
    # 初始生成100个种群
    population, population0, population1, population2, population3 = getinitialPopulation(sum(EncodeLength), L0, L1, L2,
                                                                                          L3, initialPopuSize)
    # print("population",population)
    # 最大进化代数
    maxgeneration = 80
    # 交叉概率
    prob = 0.8
    # 变异概率
    mutationprob = 0.01
    # 新生成的种群数量
    maxPopuSize = 10

    for generation in range(maxgeneration):
        # 对种群解码得到表现形
        decode, W1, V1, W2, V2 = getDecode(population, population0, population1, population2, population3, EncodeLength,
                                           decisionVariables)
        # 得到适应度值和累计概率值
        evaluation, cum_proba = getFitnessValue(fitnessFunction, decode, W1, V1, W2, V2)
        # 选择新的种群
        newpopulations = selectNewPopulation(population, cum_proba)
        # 新种群交叉
        crossPopulations = crossNewPopulation(newpopulations, prob)
        # 变异操作
        mutationpopulation = mutation(crossPopulations, mutationprob)
        # 将父母和子女合并为新的种群
        # print("mutationpopulation", mutationpopulation)
        # print("mutationpopulation.shape", mutationpopulation.shape)
        totalpopulation = np.vstack((population, mutationpopulation))
        # print("totalpopulation", totalpopulation)
        # print("totalpopulation.shape", totalpopulation.shape)
        w11, v11, w22, v22 = getPopulation(totalpopulation, totalpopulation.shape[0])

        # 最终解码
        final_decode, W11, V11, W22, V22 = getDecode(totalpopulation, w11, v11, w22, v22, EncodeLength,
                                                     decisionVariables)
        # 适应度评估
        final_evaluation, final_cumprob = getFitnessValue(fitnessFunction, final_decode, W11, V11, W22, V22)

        # 最终解码
        # final_decode = getDecode(totalpopulation,W11,V11,W22,V22, EncodeLength, decisionVariables, delta)
        # 适应度评估
        # final_evaluation, final_cumprob = getFitnessValue(fitnessFunction, final_decode)
        # 选出适应度最大的100个重新生成种群
        population = findMinPopulation(totalpopulation, final_evaluation, maxPopuSize)
        # print("pop", population)
        # 找到本轮中适应度最小的值
        optimalvalue.append(np.min(final_evaluation))
        # print("optimalvalue", optimalvalue)
        index = np.where(final_evaluation == min(final_evaluation))
        # print("index1", index)
        # print("index[0][0]", index[0][0])
        # print("pp", totalpopulation[index[0][0]])
        # print("final_decode[index[0][0]]",final_decode[index[0][0]])
        # optimalvariables.append(list(final_decode[index[0][0]]))
        # optimalvariables.append((final_decode[index[0][0]]).tolist())
        optimalvariables.append((totalpopulation[index[0][0]]).tolist())

    # print("optimalvariables", optimalvariables)

    # optimalval = np.max(optimalvalue)
    # index = np.where(optimalvalue == max(optimalvalue))
    optimalval = np.min(optimalvalue)
    #print("optimalval", optimalval)
    index = np.where(optimalvalue == min(optimalvalue))
    #print("index2", index)
    optimalvar = optimalvariables[index[0][0]]
    #print("optimalvar", optimalvar)
    optimalvar = np.array(optimalvar)
    weight11, weight21, value11, value21 = parameter_initialization(optimalvar)
    weight1, weight2, value1, value2 = getDecode1(weight11, weight21, value11, value21, decisionVariables)
    for i in range(700):
        Weight1, Weight2, Value1, Value2 = fitnessFunction(x_train, y_train, weight1, weight2, value1, value2, 3, 2, 1, 0)
    pre = test_process(x_test, y_test, Weight1, Weight2, Value1, Value2)
    print("pre\n",pre)
    print("max_y:", max(y))
    print("min_y:",min(y))
    pre_org = np.array(pre) * (max(y)-min(y)) + min(y)
    y_test_org = np.array(y_test) * (max(y) - min(y)) + min(y)
    print("pre_org\n",pre_org)
    print("y_test_org\n",y_test_org)
    errors_std = np.std(np.array(pre) - np.array(y_test))
    errors_std_org = np.std(pre_org - y_test_org)
    print("errors_std:\n",errors_std)
    print("errors_std_org\n",errors_std_org)
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