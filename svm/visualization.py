import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def view(X_train, X_test, classifier, kernel):
    r"""
    分类结果可视化
    :param X_train: 训练数据
    :param X_test: 测试数据
    :param model: SVM 分类器
    :return: None
    """
    X = np.r_[X_train, X_test]
    pred = classifier.predict(X)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    classes = np.unique(pred)

    plt.figure(figsize=(8, 6))
    for cls in classes:
        indices = pred == cls
        plt.scatter(X_2d[indices, 0], X_2d[indices, 1], label=f'Class {cls}', alpha=0.7)

    plt.title(f"Classification Results Visualization (Using Kernel {kernel})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
