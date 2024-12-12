## 模式识别作业 no.4 | 使用不同的核函数验证 SVM 算法

实验内容：在**UCI数据集**上的 Iris 和 sonar 数据上验证算法的有效性； 

数据简介：Iris 数据3类，4维，150个数据；Sonar 数据2类，60维，208个样本；

数据来源：http://archive.ics.uci.edu/ml/index.php

实验分别验证了 SVM 在线性核‘linear’、多项式核‘poly’，高斯核‘rbf’，sigmoid核上的分类效果。

### 项目结构
```
└── iris                            # iris 数据集
│   ├── iris.data                       # iris.data 数据文件
│   └── ···                             # 其他
├── sonar                           # sonar 数据集
│   ├── sonar.csv                       # sonar.csv 数据文件
│   └── ···                             # 其他
├── data_process.py                 # 数据处理文件：数据读取 / 聚类
├── svm.py                          # SVM 分类器实现，对 sklearn.svm.SVC 作二次封装
├── visualization.py                # 分类结果可视化
└── main.py                         # SVM 实验程序入口
```

### 环境配置
```
- python == 3.10
- numpy == 1.26.4
- matplotlib == 3.9.2
- pandas == 2.2.2
```

### 使用方法

- Iris 数据集
```
python main.py --dataset "iris" --C 1.0 --gamma "scale"
```
- Sonar 数据集
```
python main.py --dataset "sonar" --C 1.0 --gamma "scale"
```