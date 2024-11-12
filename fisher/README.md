## 模式识别作业 no.1 | Fisher 线性判别分析

实验内容：在**UCI数据集**上的 Iris 和 sonar 数据上验证算法的有效性； 

数据简介：Iris 数据3类，4维，150个数据；Sonar 数据2类，60维，208个样本；

数据来源：http://archive.ics.uci.edu/ml/index.php

实验代码中，采用**k折交叉验证**划分训练样本与测试样本（考虑后续加入其他划分方式的实现）

### 项目结构
```
└── iris                            # iris 数据集
│   ├── iris.data                       # iris.data 数据文件
│   └── ···                             # 其他
├── sonar                           # sonar 数据集
│   ├── sonar.csv                       # sonar.csv 数据文件
│   └── ···                             # 其他
├── data_process.py                 # 数据处理文件：数据读取 / 聚类
├── fisher_algorithm.py             # fisher 线性分类器的 python 实现
├── fisher_base.py                  # fisher 多分类器 | 二分类器的通用函数
├── fisher_iris.py                  # fisher 在 iris 多分类数据集上的具体实现（ovo & ovr）
├── fisher_sonar.py                 # fisher 在 sonar 二分类数据集上的具体实现
└── main.py                         # fisher 线性分类程序入口
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
# ovo 策略
python main.py --dataset "Iris" --expiriment "k-fold" --num_folds 10 --strategy "ovo"
# ove 策略
python main.py --dataset "Iris" --expiriment "k-fold" --num_folds 10 --strategy "ove"
```
- Sonar 数据集
```
python main.py --dataset "Sonar" --expiriment "k-fold" --num_folds 10
```