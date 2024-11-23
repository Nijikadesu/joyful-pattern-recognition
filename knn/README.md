## 模式识别作业 no.2 | 分析 K 近邻算法的错误率

实验内容：在Iris、 sonar 与 MNIST 数据集上分析 KNN 算法的错误率； 

数据简介：
- Iris 数据3类，4维，150个数据；Sonar 数据2类，60维，208个样本；
- MNIST 数据 10 类 28 * 28 单通道图片，训练集 60000 张， 测试集 10000 张。

数据来源：http://archive.ics.uci.edu/ml/index.php / torchvision.datasets.MNIST

### 项目结构
```
└── iris                            # iris 数据集
│   ├── iris.data                       # iris.data 数据文件
│   └── ···                             # 其他
├── MNSIT                           # MNIST 数据集
│   └── raw                         # .gz 格式数据
├── sonar                           # sonar 数据集
│   ├── sonar.csv                       # sonar.csv 数据文件
│   └── ···                             # 其他
├── data_process.py                 # 数据处理文件：数据读取
├── knn_algorithm.py                # KNN 分类器的 python 实现
├── main.py                         # KNN 分类器实验程序入口
└── visiualization.py               # 可视化
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
# 评估数据集规模对 KNN 错误率的影响
python main.py --dataset "iris" --num_k 5 --expiriment "change_trainset_scale"
# 评估 K 值选择对 KNN 错误率的影响
python main.py --dataset "iris" --num_k 50 --expiriment "change_num_k"
```
- Sonar 数据集
```
# 评估数据集规模对 KNN 错误率的影响
python main.py --dataset "sonar" --num_k 5 --expiriment "change_trainset_scale"
# 评估 K 值选择对 KNN 错误率的影响
python main.py --dataset "sonar" --num_k 50 --expiriment "change_num_k"
```
- MNIST 数据集
```
# 评估数据集规模对 KNN 错误率的影响
python main.py --dataset "mnist" --num_k 5 --expiriment "change_trainset_scale"
# 评估 K 值选择对 KNN 错误率的影响
python main.py --dataset "mnist" --num_k 50 --expiriment "change_num_k"
```