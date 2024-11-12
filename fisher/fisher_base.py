import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from data_process import ReadData, ClassSpliter
from fisher_algorithm import FisherClassifier, FisherPredictor

class FisherBase:
    """
    Universal functions used in FisherSonor / FisherIris
    """
    def load_data(self, data_path):
        read_data = ReadData(data_path)
        x, y, encoded_y = read_data.xy_split()

        return x, y, encoded_y

    def aggregate_data(self, x, y, encoded_y):
        class_spliter = ClassSpliter(x, y, encoded_y)
        splited_data = class_spliter.class_split()
        return splited_data

    def classifier(self, class_pos, class_neg):
        fisher_classifier = FisherClassifier(class_pos, class_neg)
        classifier = fisher_classifier.fisher()
        return classifier

    def predictor(self, classifier, class_pos, class_neg):
        predictor = FisherPredictor(
            classifier=classifier,
            class_pos=class_pos,
            class_neg=class_neg
        )
        return predictor

    def pred_check(self, pred, y, ground_truth):
        num_total = y.shape[0]
        pred = [ground_truth[p] for p in pred]
        num_correct = np.sum(pred == y)
        acc = num_correct / num_total
        return acc

    def plot_acc(self, acc):
        acc_axis = [i for i in range(len(acc))]
        plt.figure(figsize=(8, 6))
        plt.style.use('ggplot')
        plt.xlim(0, 10)
        plt.ylim(0.0, 1.0)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(acc_axis, acc, marker='o', markersize=4, markeredgewidth=2, markeredgecolor='blue')
        plt.title('Accuracy per fold')
        plt.xlabel('k-folds', fontsize=14)
        plt.ylabel('acc', fontsize=14)
        plt.show()