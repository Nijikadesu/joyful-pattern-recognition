import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KNN:
    """
    Implementation of the KNN algorithm.
    """
    def __init__(self, x_train, y_train, x_test, y_test, num_labels, k):
        """
        初始化 KNN 分类器
        :param x_train: 训练样本值
        :param y_train: 训练样本标签
        :param x_test: 测试样本值
        :param y_test: 测试样本标签
        :param num_labels: 类别数量
        :param k: 近邻数量
        """
        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_test, self.y_test = np.array(x_test), np.array(y_test)
        self.num_labels = num_labels
        self.k = k

    def dist_cal(self, x1, x2):
        """
        计算两个样本点间的距离
        :param x1: 向量1
        :param x2: 向量2
        :return: 向量间的欧氏距离
        """
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def get_k_nearest(self, x):
        """
        计算与样本 x 最近的 top K 点，并指定其中出现次数最多的标签为预测结果
        :param x: 待预测样本
        :return: 预测标记
        """
        dist_list = [0.0] * len(self.x_train)

        for i in range(len(self.x_train)):
            x_i = self.x_train[i]
            dist_list[i] = self.dist_cal(x_i, x)

        k_nearest_index = np.argsort(np.array(dist_list))[:self.k]
        return k_nearest_index

    def predict_y(self, k_nearest_index):
        """
        预测样本的标签
        :param k_nearest_index: top K 近邻的标签
        :return: 最终预测标签
        """
        label_list = [0] * self.num_labels

        for index in k_nearest_index:
            label = self.y_train[index]
            label_list[label] += 1

        pred = label_list.index(max(label_list))
        return pred

    def test(self, n_test):
        """
        计算测试集正确率
        :param n_test: 测试样本数量
        :return: 正确率
        """
        error_count = 0
        pred = []
        for i in range(n_test):
            x = self.x_test[i]
            k_nearest_index = self.get_k_nearest(x)
            y_pred = self.predict_y(k_nearest_index)
            pred.append(y_pred)
            if y_pred != self.y_test[i]:
                error_count += 1

        return (error_count / n_test), pred
