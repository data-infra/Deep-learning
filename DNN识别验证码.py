# 使用全连接神经网络类，和手写数据加载器，实现验证码识别。

import datetime
import numpy as np
import DNN   # 引入全连接神经网络
import MNIST  # 引入手写数据加载器

# 最后实现我们的训练策略：每训练10轮，评估一次准确率，当准确率开始下降时终止训练
def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = MNIST.get_training_data_set(6000,True)   # 加载训练样本数据集，和one-hot编码后的样本标签数据集
    test_data_set, test_labels = MNIST.get_test_data_set(1000,True)   # 加载测试特征数据集，和one-hot编码后的测试标签数据集
    train_data_set=np.array(train_data_set)
    train_labels=np.array(train_labels)
    test_data_set=np.array(test_data_set)
    test_labels=np.array(test_labels)

    print('样本数据集的个数：%d' % len(train_data_set))
    print('测试数据集的个数：%d' % len(test_data_set))
    network = DNN.Network([784, 300, 10],0.3)  # 定义一个输入节点784+1，神经元300，输出10，0.3为学习速率，

    while True:  # 迭代至准确率开始下降
        epoch += 1 # 记录迭代次数
        network.train(train_labels, train_data_set, 1)  # 使用训练集进行训练。1为迭代次数
        print('%s epoch %d finished' % (datetime.datetime.now(), epoch))  # 打印时间和迭代次数
        if epoch % 10 == 0:  # 每训练10次，就计算一次准确率
            error_ratio = DNN.evaluate(network, test_data_set, test_labels)  # 计算准确率
            print('%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))  # 打印输出错误率
            if error_ratio < 0.1:  # 如果错误率开始上升就不再训练了。
                break
            else:
                print('错误率：', last_error_ratio)
                last_error_ratio = error_ratio # 否则继续训练

    index=0
    for layer in network.layers:
        np.savetxt('MNIST—W'+str(index),layer.W)
        np.savetxt('MNIST—b' + str(index), layer.b)
        index+=1
        print(layer.W)
        print(layer.b)


if __name__ == '__main__':
    train_and_evaluate()   # 使用样本数据集进行预测

