import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def draw_cluster(data, labels, k, title):
    r"""
    use PCA plot resultes
    :param data: 数据集
    :param labels: 标签
    :param k: 聚类簇个数
    :param title: 图标题
    :return: None
    """
    if data.shape[-1] - 1 > 2:
        data = PCA(n_components=2).fit_transform(data)
    else:
        data = np.array(data)

    label = np.array(labels)
    plt.scatter(data[:, 0], data[:, 1], marker='o', c='black', s=7) # 原图
    colors = np.array(["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000",
                       "#008000", "#000080", "#808000", "#800080", "#008080", "#444444", "#FFD700", "#008080"])

    for i in range(k):
        plt.scatter(data[np.nonzero(label == i), 0], data[np.nonzero(label == i), 1], c=colors[i], s=7, marker='o')
    plt.title(label=title)
    plt.show()
