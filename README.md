# GA-optimized-neural-network
python 用GA算法优化BP神经网络，预测回归问题

神经网络部分：
网络结构三层：（3,2,1）

数据集：
实验的数据集为：advertise.txt （三个特征输入，一个输出）
其数据形式如下所示：（即求前三个数与最后一个数的关系）
一共有200条数据，训练集和测试集的比例为7:3

1,230.1,37.8,69.2,22.1

2,44.5,39.3,45.1,10.4

3,17.2,45.9,69.3,9.3

4,151.5,41.3,58.5,18.5

5,180.8,10.8,58.4,12.9

用GA算法优化BP神经网络的权值和阈值：
种群数量10，迭代80次，交叉概率0.8，变异概率0.01，BP神经网络学习率：0.05，迭代500次：
测试样本60个的平均无误差，errors_std_org： 1.5342603366697878
![Iamge](https://github.com/yx868868/GA-optimized-neural-network/blob/main/pic/500%E6%AC%A1.png)

迭代700次
测试样本60个的平均无误差：errors_std_org：1.0408958068854353
![Iamge](https://github.com/yx868868/GA-optimized-neural-network/blob/main/pic/700%E6%AC%A1.png)

单独用BP神经网络，学习率：0.05，迭代500次：
测试样本60个的平均无误差，errors_std_org：3.2695353501231272
![Iamge](https://github.com/yx868868/GA-optimized-neural-network/blob/main/pic/BP500.png)

单独用BP神经网络，学习率：0.05，迭代700次：
测试样本60个的平均无误差，errors_std_org：1.812
![Iamge](https://github.com/yx868868/GA-optimized-neural-network/blob/main/pic/BP700%E6%AC%A14.png)

