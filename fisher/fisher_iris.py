import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from data_process import ReadData, ClassSpliter
from fisher_base import FisherBase
from fisher_algorithm import FisherClassifier, FisherPredictor

class FisherIris(FisherBase):
    """
    Use fisher Classifier on Iris dataset.
    """
    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.exp = args.expiriment
        self.num_folds = args.num_folds
        self.strategy = args.strategy

    def ovo(self, i, j, splited_train_data, x_test):
        # one vs. one
        ground_truth = {True: i, False: j}
        classifier = self.classifier(np.array(splited_train_data[0]), np.array(splited_train_data[1]))
        predictor = FisherPredictor(classifier, np.array(splited_train_data[0]), np.array(splited_train_data[1]))
        pred = predictor.judge(x_test)
        pred = np.array([ground_truth[p] for p in pred])
        return pred

    def ove(self, i, splited_train_data, x_test):
        # one vs. rest
        ground_truth = {True: i, False: -1}
        classifier = self.classifier(np.array(splited_train_data[0]), np.array(splited_train_data[1]))
        predictor = FisherPredictor(classifier, np.array(splited_train_data[0]), np.array(splited_train_data[1]))
        pred = predictor.judge(x_test)
        pred = np.array([ground_truth[p] for p in pred])
        return pred

    def run(self):
        x, y, encoded_y = self.load_data(self.data_path)
        x, y = np.array(x), np.array(y)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        acc = []

        # k-fold 交叉验证，记录 acc
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            splited_train_data = self.aggregate_data(x_train, y_train, encoded_y)

            if self.strategy == 'ovo':
                diff_pred, final_pred = [], []
                # 训练三个分类器，记录预测结果
                for i in range(3):
                    for j in range(i + 1, 3):
                        choosed = [splited_train_data[i], splited_train_data[j]]
                        pred = self.ovo(i, j, choosed, x_test)
                        diff_pred.append(pred)

                diff_pred = np.array(diff_pred).T
                for pred in diff_pred:
                    vote = [0, 0, 0]
                    for cls in pred:
                        vote[cls] += 1
                    if np.array_equal(pred, [1, 1, 1]):
                        final_pred.append(random.randint(1, 3))
                    else:
                        final_pred.append(np.argmax(np.array(vote)))

                num_correct = np.sum(final_pred == y_test)
                num_total = len(y_test)
                acc.append(num_correct / num_total)

            elif self.strategy == 'ove':
                diff_pred, final_pred = [], []
                # 训练三个分类器， 1 vs. 2
                for i in range(3):
                    pos_train_data = splited_train_data[i]
                    neg_train_data = []
                    for j in range(3):
                        if j != i:
                            neg_train_data.extend(splited_train_data[j])
                    choosed = [pos_train_data, neg_train_data]
                    pred = self.ove(i, choosed, x_test)
                    diff_pred.append(pred)

                diff_pred = np.array(diff_pred).T
                for pred in diff_pred:
                    final_pred.append(max(pred)) # 最大值为分类结果

                num_correct = np.sum(final_pred == y_test)
                num_total = len(y_test)
                acc.append(num_correct / num_total)

            else:
                raise NotImplementedError

        print(f'{self.num_folds} fold cross-validation finished.')
        avg_acc = sum(acc) / len(acc)
        print(f'Average Valid Accuracy: {avg_acc}')
        self.plot_acc(acc)