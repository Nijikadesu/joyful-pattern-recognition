import argparse
from svm import SVM
from visualization import view
from data_process import DataProcessor
from sklearn.metrics import classification_report


def run(args):
    """
    主程序
    :param args: 命令行参数
    :return:
    """
    data_processor = DataProcessor(args.dataset)
    X_train, X_test, y_train, y_test = data_processor.load_data()

    num_labels = 0
    if args.dataset =='iris':
        num_labels += 3
    elif args.dataset == 'sonar':
        num_labels += 2

    for kernel in args.optional_kernels:
        svm_classifier = SVM(kernel=kernel, C=args.C, gamma=args.gamma)
        svm_classifier.train(X_train, y_train)
        pred = svm_classifier.predict(X_test)

        view(X_train, X_test, svm_classifier, kernel)
        Classification_Report = classification_report(y_test, pred)
        print(f"Classification Report(Using Kernel {kernel}):{Classification_Report}\n")


def get_args():
    """
    接收命令行参数
    :return: 命令行参数列表
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', "--dataset", type=str, default="sonar", help="Which dataset to train on")
    parser.add_argument('-c', "--C", type=int, default=1.0, help="regularization parameter")
    parser.add_argument('-gm', "--gamma", type=str, default="scale", help="kernel function parameter")

    args = parser.parse_args()
    args.optional_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    print("=" * 50)
    print("Running SVM algorithm on: {}".format(args.dataset))
    print("SVM regularization parameter: {}".format(args.C))
    print("SVM kernel function parameter: {}".format(args.gamma))
    print("Optional SVM kernel function: {}".format(args.optional_kernels))
    print("=" * 50)

    return args


if __name__ == '__main__':
    run(get_args())
