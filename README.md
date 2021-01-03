

# Lab1 of Web Infomation Retrieve (2020 FA of USTC)

[TOC]

## 1 安装必要的包

```
pip install simpletransformers pandas spacy networkx
```

## 2 测试（编译运行方式）

实现了 `直接分类的模型` 和 `基于实体识别+依存树+分类的模型` 两种，具体实现在下面 `关键函数的说明` 和实验报告都会介绍到。

分别运行 `train.ipynb` 中两段代码可以得到训练结果，自动保存在特定目录。

分别运行 `test.ipynb` 中两段代码可以得到预测结果，自动保存在 `outputs/` 目录。

## 3 关键函数说明

### 3.1 dataset.py

`read_train` 针对处理 train.txt 的数据为 `(sentence, (relation, src, dest))` 的格式。

`read_test` 针对处理 test.txt 的数据为 `sentence` 的格式。

### 3.2 dependency.py

- `class DepTree`

  - `shortest_path` 调用 `networkx` 得到两个实体结点在依存树中的最短路径；

  - `search` 找到实体所在位置。

### 3.3 models.py

- `class ClassifyModel` 利用 `simpletransformers.classification` 的 `ClassificationModel`

  - `train` 训练模型；

  - `predict` 预测。

- `class DependencyModel`

  - `train_ner` 训练实体模型；

  - `train_deps` 训练依存树模型（利用到训练出来的实体信息）；

  - `train` 依次调用两个训练函数；
  - `predict` 调用 `dependency.py` 中的预测函数并返回处理结果。

- `get_entity` 获取最有可能的两个实体并返回。