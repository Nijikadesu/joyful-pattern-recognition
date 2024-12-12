## 模式识别作业 no.3 | 使用不同的核函数验证 SVM 算法

实验内容：在**UCI数据集**上的 Iris 和 sonar 数据上验证算法的有效性； 

数据简介：Iris 数据3类，4维，150个数据；Sonar 数据2类，60维，208个样本；

数据来源：http://archive.ics.uci.edu/ml/index.php

实验以轮廓系数为评价指标，比较讨论了 K-Means 与 FCM 的性能差异。

### 项目结构
```
└── iris                            # iris 数据集
│   ├── iris.data                       # iris.data 数据文件
│   └── ···                             # 其他
├── sonar                           # sonar 数据集
│   ├── sonar.csv                       # sonar.csv 数据文件
│   └── ···                             # 其他
├── data_process.py                 # 数据处理文件：数据读取 / 聚类
├── kmeans.py                       # K-Means 聚类算法的 numpy 实现
├── fcm.py                          # FCM 聚类算法的 numpy 实现
├── visualization.py                # 分类结果可视化
└── main.py                         # K-Means & FCM 实验程序入口
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
python main.py --dataset "iris" --num_clusters 3 --num_rounds 50
```
- Sonar 数据集
```
python main.py --dataset "sonar" --num_clusters 2 --num_rounds 50
```