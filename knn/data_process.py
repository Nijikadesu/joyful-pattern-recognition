import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
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
        elif dataset == 'mnist':
            self.data_path = './MNIST/'

    def load_iris_or_sonar(self):
        data = pd.read_csv(self.data_path, header=None)
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

        unique_y = []
        for cls_name in y:
            if cls_name not in unique_y:
                unique_y.append(cls_name)

        encoded_y = {unique_y[i]: i for i in range(len(unique_y))}
        y = np.array([encoded_y[cls_name] for cls_name in y])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_ratio, random_state=42)

        return X_train, X_test, y_train, y_test

    def load_mnist(self):
        train_dataset = datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(),
                                       download=False)
        test_dataset = datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(),
                                      download=False)

        batch_size = len(train_dataset)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        X_train, y_train = X_train.cpu().numpy(), y_train.cpu().numpy()
        X_test, y_test = X_test.cpu().numpy(), y_test.cpu().numpy()
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)

        # 由于 MNIST 数据集规模较大，故只使用 2500 张图片作为训练集，500 张图片作为测试集
        return X_train[:2500, :], X_test[:500, :], y_train[:2500], y_test[:500]

    def load_data(self):
        if self.dataset == 'iris' or self.dataset == 'sonar':
            X_train, X_test, y_train, y_test = self.load_iris_or_sonar()
        else:
            X_train, X_test, y_train, y_test = self.load_mnist()

        return X_train, X_test, y_train, y_test
