import random
import numpy as np


class KMeans:
    """
    K-Means algorithm.
    """
    def __init__(self, dataset, num_k, max_iter=100, tol=1e-6):
        r"""
        初始化 K-means 分类器
        :param dataset: 训练数据
        :param num_k: 聚类个数 K
        :param max_iter: 最大迭代数
        :param tol: 收敛阈值
        """
        self.num_k = num_k
        self.dataset = dataset
        self.centroids = self.dataset[random.sample(range(len(self.dataset)), num_k)]
        self.max_iter = max_iter
        self.tol = tol

    def compute_distances(self):
        r"""
        计算欧式距离
        :return: 每个数据点到各聚类中心的欧式距离列表
        """
        dist_list = []
        for data in self.dataset:
            diff = np.tile(data, (self.num_k, 1)) - self.centroids
            squared_diff = diff ** 2
            squared_dist = np.sum(squared_diff, axis=1)
            dist = np.sqrt(squared_dist)
            dist_list.append(dist)
        return np.array(dist_list)

    def update_centroids(self, labels):
        r"""
        计算样本到质心的距离，更新聚类中心点
        :return: 新的聚类中心点
        """
        new_centroids = []
        for k in range(self.num_k):
            cluster_points = self.dataset[labels == k]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(self.dataset[random.randint(0, len(self.dataset) - 1)])
        return np.array(new_centroids)


    def run(self):
        r"""
        k-means 算法主函数
        :return: 聚类中心，类标签
        """
        labels = []
        for iteration in range(self.max_iter):
            distances = self.compute_distances()
            labels = np.argmin(distances, axis=1)
            new_centroids = self.update_centroids(labels)

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                # print(f"Converged at iteration {iteration}")
                break
            self.centroids = new_centroids

        return self.centroids, labels


if __name__ == '__main__':
    dataset = np.vstack((
        np.random.normal(loc=0.0, scale=1.0, size=(50, 2)),
        np.random.normal(loc=5.0, scale=1.0, size=(50, 2))
    ))
    centroids, labels = KMeans(num_k=2, dataset=dataset).run()
    print('质心为：%s' % centroids)
    print('标签分布：%s' % labels)