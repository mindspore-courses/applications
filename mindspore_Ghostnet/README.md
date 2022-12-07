# 目录

项目简介

依赖

文件组织

数据准备

# 项目简介



本项目对GhostNet网络进行了Mindspore版本的迁移，主要包含两个部分

1、对GhostNet的原理简介

2、Mindspore模型的搭建与训练

GhostNet。作者：韩凯、王云鹤等。该架构可以在同样精度下，速度和计算量均少于SOTA算法。该论文提供了一个全新的Ghost模块，旨在通过廉价操作生成更多的特征图。基于一组原始的特征图，作者应用一系列线性变换，以很小的代价生成许多能从原始特征发掘所需信息的“幻影”特征图（Ghost feature maps）。该Ghost模块即插即用，通过堆叠Ghost模块得出Ghost bottleneck，进而搭建轻量级神经网络——GhostNet。在ImageNet分类任务，GhostNet在相似计算量情况下**Top-1正确率达75.7%**，高于MobileNetV3的75.2%。

# 依赖

Python 3.0+

Mindspore 1.8.1+

# 文件组织

```
mindspore_Ghostnet
├── CIFAR10        //数据集
├── GhostNet.ipynb //ipynb文件
└── logXXXXXX.TXT  //记录文件
```

# 数据准备

这里使用了 ***download.download函数*** 来下载 ***CIFAR-10*** 数据集

CIFAR-10数据集由60000张32x32彩色图片组成，总共有10个类别，每类6000张图片。有50000个训练样本和10000个测试样本。10个类别包含飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车。 整个数据集被分为5个训练批次和1个测试批次，每一批10000张图片。测试批次包含10000张图片，是由每一类图片随机抽取出1000张组成的集合。剩下的50000张图片每一类的图片数量都是5000张，训练批次是由剩下的50000张图片打乱顺序，然后随机分成5份，所以可能某个训练批次中10个种类的图片数量不是对等的，会出现一个类的图片数量比另一类多的情况。

```python
# 数据集根目录
data_dir = "./CIFAR10"

# 下载解压并加载CIFAR-10训练数据集
download_train = Cifar10(path=data_dir, split="train", batch_size=4096, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

step_size = dataset_train.get_dataset_size()#TODO

# 下载解压并加载CIFAR-10测试数据集
# dataset_val = Cifar10(path=data_dir, split='test', batch_size=6, resize=32, download=True)
download_eval = Cifar10(path=data_dir, split="test", batch_size=1024, resize=32, download=True)
dataset_eval = download_eval.run()
```