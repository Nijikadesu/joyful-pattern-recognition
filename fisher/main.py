import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fisher_sonar import FisherSonar
from fisher_iris import FisherIris

def run(args):
    if args.dataset == "Sonar":
        args.data_path = "../dataset/sonar/sonar.csv"
        FisherSonar(args).run()
    elif args.dataset == "Iris":
        args.data_path = "../dataset/iris/iris.data"
        FisherIris(args).run()
    else:
        raise NotImplementedError

def get_args():
    """
    接收命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', "--dataset", type=str, default="Sonar", help="Which dataset to train on")
    parser.add_argument('-path', "--data_path", type=str, default="./sonar/sonar.csv", help="Path to the dataset")
    parser.add_argument('-exp', "--expiriment", type=str, default="k-fold", help="Expiriment settings")
    parser.add_argument('-fold', "--num_folds", type=int, default=10, help="Number of folds K")
    parser.add_argument('-str', "--strategy", type=str, default='ovo', help="spliting strategy")

    args = parser.parse_args()

    print("=" * 50)
    print("Running fisher classification on: {}".format(args.dataset))
    print("Evaluating model performance with: {}".format(args.expiriment))
    print("k = {}".format(args.num_folds))
    if args.dataset == "Iris":
        print("Using strategy: {}".format(args.strategy))
    print("=" * 50)

    return args

if __name__ == '__main__':
    run(get_args())