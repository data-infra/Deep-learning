# 实现神经网络反向传播算法，以此来训练网络。全连接神经网络可以包含多层，但是只有最后一层输出前有激活函数。
# 所谓向量化编程，就是使用矩阵运算。

import random
import math
import numpy as np
import datetime
import Activators  # 引入激活器模块

# 1. 当为array的时候，默认d*f就是对应元素的乘积，multiply也是对应元素的乘积，dot（d,f）会转化为矩阵的乘积， dot点乘意味着相加，而multiply只是对应元素相乘，不相加
# 2. 当为mat的时候，默认d*f就是矩阵的乘积，multiply转化为对应元素的乘积，dot（d,f）为矩阵的乘积



# 全连接每层的实现类。输入对象x、神经层输出a、输出y均为列向量
class FullConnectedLayer(object):
    # 全连接层构造函数。input_size: 本层输入列向量的维度。output_size: 本层输出列向量的维度。activator: 激活函数
    def __init__(self, input_size, output_size,activator,learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W。初始化权重为(rand(output_size, input_size) - 0.5) * 2 * sqrt(6 / (output_size + input_size))
        wimin = (output_size - 0.5) * 2 * math.sqrt(6 / (output_size + input_size))
        wimax = (input_size-0.5)*2*math.sqrt(6/(output_size + input_size))
        # self.W = np.random.uniform(wimin,wimax,(output_size, input_size))  #初始化为-0.1~0.1之间的数。权重的大小。行数=输出个数，列数=输入个数。a=w*x，a和x都是列向量
        self.W = np.random.uniform(-0.1, 0.1,(output_size, input_size))  # 初始化为-0.1~0.1之间的数。权重的大小。行数=输出个数，列数=输入个数。a=w*x，a和x都是列向量
        # 偏置项b
        self.b = np.zeros((output_size, 1))  # 全0列向量偏重项
        # 学习速率
        self.learning_rate = learning_rate
        # 输出向量
        self.output = np.zeros((output_size, 1)) #初始化为全0列向量

    # 前向计算，预测输出。input_array: 输入列向量，维度必须等于input_size
    def forward(self, input_array):   # 式2
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    # 反向计算W和b的梯度。delta_array: 从上一层传递过来的误差项。列向量
    def backward(self, delta_array):
        # 式8
        self.delta = np.multiply(self.activator.backward(self.input),np.dot(self.W.T, delta_array))   #计算当前层的误差，以备上一层使用
        self.W_grad = np.dot(delta_array, self.input.T)   # 计算w的梯度。梯度=误差.*输入
        self.b_grad = delta_array  #计算b的梯度

    # 使用梯度下降算法更新权重
    def update(self):
        self.W += self.learning_rate * self.W_grad
        self.b += self.learning_rate * self.b_grad


# 神经网络类
class Network(object):
    # 初始化一个全连接神经网络。layers:数组，描述神经网络每层节点数。包含输入层节点个数、隐藏层节点个数、输出层节点个数
    def __init__(self, layers,rate):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i+1],Activators.SigmoidActivator(),rate))   # 创建全连接层，并添加到layers中


    # 训练函数。labels: 样本标签矩阵。data_set: 输入样本矩阵。epoch: 训练轮数
    def train(self, labels, data_set,epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                # print(i,'次迭代，',d,'个样本')
                oneobject = np.array(data_set[d]).reshape(-1,1)   #将输入对象和输出标签转化为列向量
                onelabel = np.array(labels[d]).reshape(-1,1)
                self.train_one_sample(onelabel,oneobject)

    # 内部函数，用一个样本训练网络
    def train_one_sample(self, label, sample):
        # print('样本：\n',sample)
        self.predict(sample)  # 根据样本对象预测值
        self.calc_gradient(label) # 计算梯度
        self.update_weight() # 更新权重

    # 使用神经网络实现预测。sample: 输入样本
    def predict(self, sample):
        sample = sample.reshape(-1,1)   #将样本转换为列向量
        output = sample  # 输入样本作为输入层的输出
        for layer in self.layers:
            # print('权值：',layer.W,layer.b)
            layer.forward(output)  # 逐层向后计算预测值。因为每层都是线性回归
            output = layer.output
        # print('预测输出：', output)
        return output

    # 计算每个节点的误差。label为一个样本的输出向量，也就对应了最后一个所有输出节点输出的值
    def calc_gradient(self, label):
        # print('计算梯度：',self.layers[-1].activator.backward(self.layers[-1].output).shape)
        delta = np.multiply(self.layers[-1].activator.backward(self.layers[-1].output),(label - self.layers[-1].output))  #计算输出误差
        # print('输出误差：', delta.shape)
        for layer in self.layers[::-1]:
            layer.backward(delta)   # 逐层向前计算误差。计算神经网络层和输入层误差
            delta = layer.delta
            # print('当前层误差：', delta.shape)
        return delta

    # 更新每个连接权重
    def update_weight(self):
        for layer in self.layers:  # 逐层更新权重
            layer.update()


# ====================================以上为网络的类构建=================================
# ====================================以下为网络的应用=================================


# 根据返回结果计算所属类型
def valye2type(vec):
    return vec.argmax(axis=0)   # 获取概率最大的分类，由于vec是列向量，所以这里按列取最大的位置

# 使用错误率来对网络进行评估
def evaluate(network, test_data_set, test_labels):
    error = 0
    total = test_data_set.shape[0]
    for i in range(total):
        label = valye2type(test_labels[i])
        predict = valye2type(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)




# 由于使用了逻辑回归函数，所以只能进行分类识别。识别ont-hot编码的结果
if __name__ == '__main__':
    # 使用神经网络实现and运算
    data_set = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[1,0],[1,0],[1,0],[0,1]])
    # print(data_set)
    # print(labels)
    net = Network([2,1,2],1)  # 输入节点2个（偏量b会自动加上），神经元1个，输出节点2个。rate: 学习速率为1
    net.train(labels, data_set,100)
    for layer in net.layers:  # 网络层总不包含输出层
        print('W:',layer.W)
        print('b:',layer.b)

    # 对结果进行预测
    sample = np.array([[1,1]])
    result = net.predict(sample)
    type = valye2type(result)
    print('分类：',type)



#==============================================向量化编程实现全连接网络，实现调制方式识别========================================

#
# # 最后实现我们的训练策略：每训练10轮，评估一次准确率，当准确率开始下降时终止训练
# def test():
#     # 产生1,1附近的数据
#     s01 = np.random.normal(1, 0.3, 1000).reshape((1000, 1))
#     s02 = np.random.normal(1, 0.3, 1000).reshape((1000, 1))
#     type0 = np.zeros((s01.shape[0],4))
#     type0[:,0]=1
#     s0 = np.column_stack((s01, s02))
#     # 产生-1,1附近的数据
#     s11 = np.random.normal(-1, 0.3, 1000).reshape((1000, 1))
#     s12 = np.random.normal(1, 0.3, 1000).reshape((1000, 1))
#     type1 = np.zeros((s11.shape[0],4))
#     type1[:,1]=1
#     s1 = np.column_stack((s11, s12))
#     # 产生-1，-1附近的数据
#     s21 = np.random.normal(-1, 0.3, 1000).reshape((1000, 1))
#     s22 = np.random.normal(-1, 0.3, 1000).reshape((1000, 1))
#     type2 = np.zeros((s21.shape[0],4))
#     type2[:,0]=1
#     s2 = np.column_stack((s21, s22))
#     # 产生1，-1附近的数据
#     s31 = np.random.normal(1, 0.3, 1000).reshape((1000, 1))
#     s32 = np.random.normal(-1, 0.3, 1000).reshape((1000, 1))
#     type3 = np.zeros((s31.shape[0],4))
#     type3[:,0]=1
#     s3 = np.column_stack((s31, s32))
#
#
#     train_data_set = np.row_stack((s0,s1,s2,s3))   # 属性数据集合并
#     train_labels=np.row_stack((type0,type1,type2,type3))  # 标签数据集合并
#
#     last_error_ratio = 1.0
#     epoch = 0
#
#     network = Network([2, 3, 4])  # 定义一个输入节点784+1，神经元300，输出10
#
#     while True:  # 迭代至准确率开始下降
#         epoch += 1 # 记录迭代次数
#         network.train(train_labels, train_data_set, 0.3, 1)  # 使用训练集进行训练。0.3为学习速率，1为迭代次数
#         print('%s epoch %d finished' % (datetime.datetime.now(), epoch))  # 打印时间和迭代次数
#         if epoch % 10 == 0:  # 每训练10次，就计算一次准确率
#             error_ratio = evaluate(network, train_data_set, train_labels)  # 计算准确率
#             print('%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))  # 打印输出错误率
#             if error_ratio >= last_error_ratio:  # 如果错误率开始上升就不再训练了。
#                 break
#             else:
#                 last_error_ratio = error_ratio # 否则继续训练
#
#     # 将网络固化到文本中，以便以后使用
#     print('逐层打印：\n')
#     for layer in network.layers:
#         print(layer.W)
#         print(layer.b)