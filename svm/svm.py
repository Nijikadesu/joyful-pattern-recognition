from sklearn.svm import SVC


class SVM:
    def __init__(self, kernel, C=1.0, gamma='scale'):
        r"""
        初始化 SVM
        :param kernel: 核函数选择
        :param C: 正则化系数
        :param gamma: 核函数参数
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, X_train, y_train):
        r"""
        拟合训练集
        :param X_train: 训练数据
        :param y_train: 训练样本标签
        :return: None
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        r"""
        在测试集上进行预测
        :param X_test: 测试数据
        :return: 预测结果
        """
        return self.model.predict(X_test)
