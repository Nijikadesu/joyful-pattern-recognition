import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Load Data.
    """
    def __init__(self, dataset, split_ratio=0.2):
        '''
        初始化数据加载器
        :param dataset: 数据集名称
        '''
        self.dataset = dataset
        self.data_path = ''
        self.split_ratio = split_ratio
        if dataset == 'iris':
            self.data_path = './iris/iris.data'
        elif dataset == 'sonar':
            self.data_path = './sonar/sonar.csv'

    def load_data(self):
        r"""
        加载数据集
        :return: 训练集与测试集
        """
        data = pd.read_csv(self.data_path, header=None)

        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

        # encode y
        unique_y = []
        for cls_name in y:
            if cls_name not in unique_y:
                unique_y.append(cls_name)

        encoded_y = {unique_y[i]: i for i in range(len(unique_y))}
        y = np.array([encoded_y[cls_name] for cls_name in y])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_ratio, random_state=42)

        # standarize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
