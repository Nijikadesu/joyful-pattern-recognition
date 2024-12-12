import numpy as np
from tqdm.auto import tqdm


class FCM:
    """
    FCM algorithm
    """
    def __init__(self, dataset, num_c, m=2.0, max_iter=100, tol=1e-5):
        r"""
        初始化 FCM 算法
        :param dataset: 数据集
        :param num_c: 聚类簇个数
        :param m: 模糊度因子
        :param max_iter: 最大迭代数
        :param tol: 收敛阈值
        """
        self.dataset = dataset
        self.num_samples, self.num_features = dataset.shape
        self.m = m
        self.num_c = num_c
        self.max_iter = max_iter
        self.tol = tol

        self.U = self.initialize_membership_matrix()

    def initialize_membership_matrix(self):
        r"""
        初始化隶属度矩阵 U。
        :return: 隶属度矩阵 U
        """
        U = np.random.rand(self.num_samples, self.num_c)
        U = U / U.sum(axis=1, keepdims=True)
        return U

    def update_cluster_centers(self):
        r"""
        根据隶属度矩阵更新聚类中心。
        :return: 聚类中心
        """
        um = self.U ** self.m
        centroids = (um.T @ self.dataset) / um.sum(axis=0)[:, None]
        return centroids

    def update_membership_matrix(self, centroids):
        r"""
        根据公式更新隶属度矩阵 U。
        :return: 更新后的隶属度矩阵
        """
        dist = np.linalg.norm(self.dataset[:, None, :] - centroids[None, :, :], axis=2)
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        inv_dist = 1 / dist
        U_new = inv_dist ** (2 / (self.m - 1))
        U_new = U_new / U_new.sum(axis=1, keepdims=True)
        return U_new

    def run(self):
        r"""
        模糊 c 均值聚类实现。
        :return:
        """
        centroids, labels = [], []
        for iteration in range(self.max_iter):
            U_old = self.U.copy()
            centroids = self.update_cluster_centers()
            self.U = self.update_membership_matrix(centroids)
            if np.linalg.norm(self.U - U_old) < self.tol:
                # print(f"Converged at iteration {iteration}")
                break

            labels = np.argmax(self.U, axis=1)

        return centroids, labels


if __name__ == "__main__":
    # 创建数据集
    np.random.seed(42)
    data = np.vstack((
        np.random.normal(loc=0.0, scale=1.0, size=(50, 2)),
        np.random.normal(loc=5.0, scale=1.0, size=(50, 2))
    ))
    n_clusters = 2
    centers, labels = FCM(data, n_clusters).run()
    print("聚类中心：")
    print(centers)
    print("\n类标签：")
    print(labels)
