import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from data_process import ReadData, ClassSpliter
from fisher_base import FisherBase
from fisher_algorithm import FisherClassifier, FisherPredictor

class FisherSonar(FisherBase):
    """
    Use fisher Classifier on Sonar dataset.
    """
    def __init__(self, args):
        super().__init__() # 继承父类 FisherBase
        self.data_path = args.data_path
        self.exp = args.expiriment
        self.num_folds = args.num_folds

    def run(self):
        x, y, encoded_y = self.load_data(self.data_path)
        x, y = np.array(x), np.array(y)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        ground_truth = {True: 0, False:1} # 类标记为0的类作为正类，1作为负类
        acc = [] # 记录精度

        # k-fold 交叉验证，记录 acc
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            splited_train_data = self.aggregate_data(x_train, y_train, encoded_y)

            # 设置 train_data 第 0 列为正类, 第 1 列为负类
            classifier = self.classifier(np.array(splited_train_data[0]), np.array(splited_train_data[1]))
            predictor = self.predictor(classifier, np.array(splited_train_data[0]), np.array(splited_train_data[1]))

            pred = predictor.judge(x_test)
            acc.append(self.pred_check(pred, y_test, ground_truth))

        print(f'{self.num_folds} fold cross-validation finished.')
        avg_acc = sum(acc) / len(acc)
        print(f'Average Valid Accuracy: {avg_acc}')
        self.plot_acc(acc)