import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns


class Visualizer:
    """
    实验结果的可视化
    """
    @staticmethod
    def draw_plot(acc, ax, exp):
        plt.figure(figsize=(8, 6))
        plt.plot(ax, acc, label='train acc', marker='o', linestyle='-', color='b')

        plt.title('Error Curve', fontsize=16)
        if exp == "change_num_k":
            plt.xlabel('num k', fontsize=14)
        else:
            plt.xlabel('train ratio', fontsize=14)
        plt.ylabel('error rate', fontsize=14)

        plt.legend()
        plt.grid(True)
        plt.show()
