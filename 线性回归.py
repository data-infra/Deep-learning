# -*- coding: UTF-8 -*-
# 线性回归完成and运算
from functools import reduce

class Perceptron(object):
    # 初始化感知器，设置输入参数的个数，以及激活函数。
    # 激活函数的类型为double -> double
    def __init__(self, input_num, activator):
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    # 打印学习到的权重、偏置项
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    # 输入向量，输出感知器的计算结果
    def predict(self, input_vec):
        sum=0
        for i in range(len(input_vec)):
            sum+=input_vec[i]*self.weights[i]
        predicted=self.activator(sum+self.bias)
        return predicted

    # 输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    # 一次迭代，把所有的训练数据过一遍
    def _one_iteration(self, input_vecs, labels, rate):
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)
    # 按照感知器规则更新权重
    def _update_weights(self, input_vec, output, label, rate):
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        for i in range(len(input_vec)):
            self.weights[i] = self.weights[i] + rate * delta * input_vec[i]
        # 更新bias
        self.bias += rate * delta








# ==================使用测试=======================

def activfun(x):
    return 1 if x > 0 else 0

#使用and真值表训练感知器
def train_and_perceptron():
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, activfun)
    # 设置一个and运算的样本数据集
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    # 训练，迭代10轮, 学习速率为0.1
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))