import numpy as np
import pandas as pd

iris_path = '../dataset/iris/iris.data'
sonar_path = '../dataset/sonar/sonar.csv'

class ReadData:
    """
    Read data from dataset files.
    """
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, header=None)

    def xy_split(self):
        x = self.data.iloc[:, :-1].to_numpy()
        y = self.data.iloc[:, -1].to_numpy()

        unique_y = []
        for cls_name in y:
            if cls_name not in unique_y:
                unique_y.append(cls_name)

        encoded_y = {unique_y[i]: i for i in range(len(unique_y))}
        y = [encoded_y[cls_name] for cls_name in y]

        return x, y, encoded_y

class ClassSpliter:
    """
    Split data by class name.
    """
    def __init__(self, x, y, encoded_y):
        self.x = x
        self.y = y
        self.encoded_y = encoded_y

    def class_split(self):
        num_samples = len(self.y)
        num_classes = len(self.encoded_y)
        splited_data = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            class_idx = self.y[i]
            splited_data[class_idx].append(self.x[i])
        return splited_data

if __name__ == '__main__':
    # 数据划分验证
    read_data = ReadData(sonar_path)
    x, y, encoded_y = read_data.xy_split()
    print(x[:5, :], y[:5])
    class_spliter = ClassSpliter(x, y, encoded_y)
    splited_data = class_spliter.class_split()
    print(len(splited_data))
    for i in range(len(splited_data)):
        print(len(splited_data[i]))
    print(splited_data[0][0])
    print(encoded_y)