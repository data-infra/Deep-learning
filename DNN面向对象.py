# 实现神经网络反向传播算法，以此来训练网络
# 使用面向对象的编程方法

from functools import reduce
import random
from numpy import *

# sigmoid函数，逻辑回归函数，将线性回归值转化为概率的激活函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index  #节点所属的层的编号
        self.node_index = node_index  #节点的编号
        self.downstream = []   #下游连接
        self.upstream = []  #上游连接
        self.output = 0   #输出值
        self.delta = 0   #误差值

    # 设置节点的输出值。如果节点属于输入层会用到这个函数。
    def set_output(self, output):
        self.output = output

    # 添加一个到下游节点的连接
    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    # 添加一个到上游节点的连接
    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    # 根据式1计算节点的输出
    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    # 节点属于隐藏层时，根据式4计算delta
    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    # 节点属于输出层时，根据式3计算delta
    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    # 打印节点的信息
    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

# ConstNode对象，为了实现每一层的一个输出恒为1的节点(计算偏置项时需要)
class ConstNode(object):
    # 构造偏量b的节点对象。
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    # 添加一个到下游节点的连接
    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    # 节点属于隐藏层时，根据式4计算delta
    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    # 打印节点的信息
    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


# Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作
class Layer(object):
    # 初始化一层。layer_index:层编号。node_count:层所包含的节点个数
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    # 设置层的输出。当层是输入层时会用到。
    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    # 计算层的输出向量
    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    # 打印层的信息
    def dump(self):
        for node in self.nodes:
            print
            node


# Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
class Connection(object):
    # 初始化连接，权重初始化为是一个很小的随机数
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node   #连接的上游节点
        self.downstream_node = downstream_node   #连接的下游节点
        self.weight = random.uniform(-0.1, 0.1)  # 权重值。
        self.gradient = 0.0   # 梯度，在更新权重时使用

    # 计算梯度
    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    # 根据梯度下降算法更新权重
    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    # 获取当前的梯度
    def get_gradient(self):
        return self.gradient

    # 打印连接信息
    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

# Connections对象，提供Connection集合操作。
class Connections(object):
    def __init__(self):
        self.connections = []

    # 添加链接
    def add_connection(self, connection):
        self.connections.append(connection)

    # 打印信息
    def dump(self):
        for conn in self.connections:
            print
            conn

# 封装以上所有类，形成网络类。提供API。
class Network(object):
    # 初始化一个全连接神经网络。layers:数组，描述神经网络每层节点数
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))  # 添加每个网络层，（层次和节点数目）
        for layerindex in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layerindex].nodes
                           for downstream_node in self.layers[layerindex + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    # 训练神经网络。labels: 数组，训练样本的标签。每个元素是一个样本的标签。data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。rate为学习速率，epoch为迭代次数
    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):  #遍历每一个样本对象
                self.train_one_sample(labels[d], data_set[d], rate)  # 根据每一个样本对象（多特征，多输出）训练网络
                print('样本 %d 训练结束' % d)

    # 内部函数，用一个样本训练网络
    def train_one_sample(self, label, sample, rate):
        self.predict(sample)  # 根据样本对象预测值
        self.calc_delta(label)  # 计算误差
        self.update_weight(rate)  # 更新权重

    # 内部函数，计算每个节点的误差。label为一个样本的输出向量，也就对应了最后一个所有输出节点输出的值
    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes  # 最后一层为输出节点
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])  # 计算输出层节点的误差。每一个节点对应一个维度的值
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()   # 计算隐藏层和输入层节点的误差

    # 内部函数，更新每个连接权重
    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    # 内部函数，计算每个连接的梯度
    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    # 获得网络在一个样本下，每个连接上的梯度。label: 样本标签，sample: 样本输入
    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    # 根据输入的样本预测输出值。sample: 数组，样本的特征，也就是网络的输入向量
    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    # 打印网络信息
    def dump(self):
        for layer in self.layers:
            layer.dump()





def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                            )
                        )

# 梯度检查。network: 神经网络对象，sample_feature: 样本的特征，sample_label: 样本的标签
def gradient_check(network, sample_feature, sample_label):
    # 计算网络误差
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (expected_gradient, actual_gradient))




# 数据生成器
class Normalizer(object):
    def __init__(self):
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        arr = []
        for item in self.mask:  # mask数组每个元素与输入数字按位与。
            if number & item:
                arr.append(0.9)
            else:
                arr.append(0.1)
        return arr

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


# 获取训练数据集。这里的特征只使用一个
def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(32):   # 产生32个样本
        randtest = int(random.uniform(0, 256))  # 随机产生一个数
        n = normalizer.norm(randtest)  #产生一个数组
        data_set.append(n)  # 数组作为样本属性
        labels.append(n)  # 数组也作为样本标签
    return labels, data_set

# 训练全连接网络。
def train(network):
    labels, data_set = train_data_set()   # 获取样本数据集
    print(list(data_set))
    print(list(labels))
    network.train(labels, data_set, 0.3, 10)

#使用网络测试新数据
def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (data, normalizer.denorm(predict_data)))

# 纠正率
def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])  # 输入节点8个，神经元3个，输出节点8个。数据集中就是8个特征的样本数据集
    train(net)  # 训练网络
    # net.dump()  # 打印输出
    # correct_ratio(net)