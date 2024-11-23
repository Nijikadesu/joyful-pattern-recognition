import math
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from knn_algorithm import KNN
from visualization import Visualizer
from data_process import DataProcessor

def run(args):
    """
    主程序
    :param args: 命令行参数
    :return:
    """
    data_processor = DataProcessor(args.dataset)
    X_train, X_test, y_train, y_test = data_processor.load_data()

    num_labels = 0
    if args.dataset == 'mnist':
        num_labels += 10
    elif args.dataset == 'iris':
        num_labels += 3
    elif args.dataset == 'sonar':
        num_labels += 2

    if args.expiriment == "change_num_k":
        acc = []
        ax = np.arange(0, args.num_k + 1, 1)
        num_test = int(len(y_test) * args.test_ratio)
        for k in tqdm(ax):
            knn = KNN(X_train, y_train, X_test, y_test, num_labels, k)
            ac, pred = knn.test(num_test)
            acc.append(ac)
        print("epiriment result:")
        print(f"mean: {np.mean(acc)}, std-var: {np.var(acc)}")
        Visualizer().draw_plot(acc, ax, args.expiriment)
    elif args.expiriment == "change_trainset_scale":
        acc = []
        num_test = int(len(y_test) * args.test_ratio)
        ax = np.arange(0.00, 1.00, 0.02)
        for ratio in tqdm(ax):
            num_train = int(y_train.shape[0] * ratio)
            X_subtrain, y_subtrain = X_train[:num_train], y_train[:num_train]
            knn = KNN(X_subtrain, y_subtrain, X_test, y_test, num_labels, args.num_k)
            ac, pred = knn.test(num_test)
            acc.append(ac)
        print("epiriment result:")
        print(f"mean: {np.mean(acc)}, std-var: {np.var(acc)}")
        Visualizer().draw_plot(acc, ax, args.expiriment)


def get_args():
    """
    接收命令行参数
    :return: 命令行参数列表
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', "--dataset", type=str, default="mnist", help="Which dataset to train on")
    parser.add_argument('-k', "--num_k", type=int, default="5", help="(Max) Number of nerghbors")
    parser.add_argument('-exp', "--expiriment", type=str, default="change_trainset_scale", help="Expiriment settings")
    parser.add_argument('-train_r', "--train_ratio", type=float, default="1.0", help="Ratio of train data used")
    parser.add_argument('-test_r', "--test_ratio", type=float, default="0.2", help="Ratio of test data used")

    args = parser.parse_args()

    print("=" * 50)
    print("Running KNN algorithm on: {}".format(args.dataset))
    print("Max number of neighbors: {}".format(args.num_k))
    print("Ratio of train data used: {}".format(args.train_ratio))
    print("Ratio of test data used: {}".format(args.test_ratio))
    if args.expiriment == "change_trainset_scale":
        print("\nIn this expiriment, \nwe evaluate the impact of training set scale on the KNN algorithm \nby changing the number of K from 0 to 50, \nstep_size = 1.")
    else:
        print("\nIn this expiriment, \nwe evaluate the impact of K value changes on the KNN algorithm \nby changing the percentage of training data used from 0.00 to 1.00, \nstep_size = 0.02.")
    print("=" * 50)

    return args

if __name__ == '__main__':
    run(get_args())
