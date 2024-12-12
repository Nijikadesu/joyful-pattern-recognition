import argparse
import numpy as np
from fcm import FCM
from kmeans import KMeans
from data_process import DataProcessor
from visualization import draw_cluster
from sklearn.metrics import silhouette_score


def calculate_silhouette_score(dataset, labels):
    """
    计算轮廓系数 (Silhouette Score)
    :param dataset: 数据集
    :param labels: 每个数据点的聚类标签
    :return: 轮廓系数
    """
    return silhouette_score(dataset, labels)

def expiriment(args, dataset):
    kmeans_scores, fcm_scores = [], []
    k_labels, c_labels = 0, 0
    for i in range(args.num_rounds):
        # print("Evaluating on K-means...")
        k_centroids, k_labels = KMeans(dataset=dataset, num_k=args.num_clusters).run()
        kmeans_score = calculate_silhouette_score(dataset, k_labels)

        # print("Evaluating on FCM...")
        c_centroids, c_labels = FCM(dataset=dataset, num_c=args.num_clusters).run()
        fcm_score = calculate_silhouette_score(dataset, c_labels)

        kmeans_scores.append(kmeans_score)
        fcm_scores.append(fcm_score)

    score1 = np.array(kmeans_scores).mean()
    score2 = np.array(fcm_scores).mean()

    print(f"KMeans 平均轮廓系数: {score1}")
    print(f"FCM 平均轮廓系数: {score2}")

    draw_cluster(dataset, labels=k_labels, k=args.num_clusters, title=f"k-means on {args.dataset}")
    draw_cluster(dataset, labels=c_labels, k=args.num_clusters, title=f"fcm on {args.dataset}")

def run(args):
    """
    主程序
    :param args: 命令行参数
    :return:
    """
    data_processor = DataProcessor(args.dataset)
    dataset, _ = data_processor.load_data()
    # print(f"Using Dataset {args.dataset}.")
    expiriment(args, dataset)

def get_args():
    """
    接收命令行参数
    :return: 命令行参数列表
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', "--dataset", type=str, default="iris", help="Which dataset to train on")
    parser.add_argument('-nc', "--num_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument('-ne', "--num_rounds", type=int, default=50, help="Number of expiriment rounds")
    args = parser.parse_args()

    print("=" * 50)
    print("Running algorithm on: {}".format(args.dataset))
    print("Number of clusters: {}".format(args.num_clusters))
    print("Number of expiriment rounds: {}".format(args.num_rounds))
    print("=" * 50)

    return args

if __name__ == '__main__':
    run(get_args())
