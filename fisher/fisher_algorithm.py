import math
import random
import numpy as np
import matplotlib.pyplot as plt
from data_process import ReadData, ClassSpliter

class FisherClassifier:
    """
    Implementation of fisher Binary Classifier.
    """
    def __init__(self, class_pos, class_neg):
        self.class_pos = class_pos
        self.class_neg = class_neg

    def cal_cov_and_avg(self, samples):
        samples = samples.astype(float)
        avg = np.mean(samples, axis=0)
        cov = np.zeros((samples.shape[1], samples.shape[1]))
        for sample in samples:
            std_var = (sample - avg).reshape(-1, 1)
            cov += np.matmul(std_var, std_var.T)
        return cov, avg

    def fisher(self):
        cov_1, avg_1 = self.cal_cov_and_avg(self.class_pos)
        cov_2, avg_2 = self.cal_cov_and_avg(self.class_neg)
        S_w = cov_1 + cov_2
        # 利用奇异值分解求取矩阵的逆
        u, s, v = np.linalg.svd(S_w)
        S_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
        w = np.matmul(S_w_inv, avg_1 - avg_2)
        return w

class FisherPredictor:
    """
    Predict classes of given-samples using fisher Classifier.
    """
    def __init__(self, classifier, class_pos, class_neg):
        self.num_features = classifier.shape[0]
        self.classifier = classifier.reshape(1, -1)
        self.class_pos = class_pos
        self.class_neg = class_neg

    def judge(self, samples):
        samples = samples.reshape(-1, self.num_features, 1)
        avg_1 = np.mean(self.class_pos, axis=0).reshape(-1, 1)
        avg_2 = np.mean(self.class_neg, axis=0).reshape(-1, 1)
        center_1 = np.dot(self.classifier, avg_1)
        center_2 = np.dot(self.classifier, avg_2)
        pos = np.dot(self.classifier, samples)
        return np.array(abs(pos-center_1) < abs(pos-center_2)).flatten()

if __name__ == '__main__':
    # 验证 fisher 分类器有效性
    iris_path = '../dataset/iris/iris.data'
    sonar_path = '../dataset/sonar/sonar.csv'

    read_data = ReadData(sonar_path)
    x, y, encoded_y = read_data.xy_split()
    class_spliter = ClassSpliter(x, y, encoded_y)
    splited_data = class_spliter.class_split()

    fisher_classifier = FisherClassifier(class_pos=np.array(splited_data[0]),
                                         class_neg=np.array(splited_data[1]))

    classifier = fisher_classifier.fisher()
    print(classifier.shape)

    sample = np.array(splited_data[0][:20])
    fisher_predictor = FisherPredictor(
        classifier=classifier,
        class_pos = splited_data[0], # 设置 R 为正类
        class_neg = splited_data[1]  # 设置 M 为负类
    )

    pred = fisher_predictor.judge(sample)
    y = np.array([0] * 20)
    print(np.sum((pred == y)))