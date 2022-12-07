# 基于MindSpore框架的Swin Transformer案例实现

## 1 模型简介

### 1.1 模型背景

Swin Transfromer在2021年首次发表于论文《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》，目前已用于图像分割、分类等计算机视觉领域的各项任务中。该模型借鉴了Vision Transformer模型的思想，将二维图像加工成transformer可处理的一维数据，试图将transformer这一自然语言处理领域的利器，迁移至计算机视觉领域，以获得较优的模型性能。

目前，transformer应用到CV领域，需要克服一些难题：

- 随着图像的分辨率增加，展平后的序列长度呈平方级别增加，是模型不可接受的，将严重影响transformer中自注意力的计算效率；
- 不同于NLP模型的传统输入特征，同一物体的图像因拍摄角度等原因，尺度和内容特征方差较大。

Swin Transformer创新地引入了滑动窗口机制，在窗口内进行自注意力计算，使计算复杂度随图片分辨率平方级别增长降低为线性增长，并使模型可以学习到跨窗口的图像信息；参考了传统的CNN结构，进行类似的级联式局部特征提取工作，在每一层次进行下采样以扩大感受野，并可提取到多尺度的图像信息。

### 1.2 模型基本结构

Swin Transfromer的基本结构如图1(d)所示，由4个层级式模块组成，每一模块内包含一个Swin Transformer Block。原始输入图像尺寸为H×W×3（3为RGB通道数），经过Patch Partition层分为大小为4×4的patch，尺寸转换为(H/4)×(W/4)×48，其功能等价于卷积核大小为4×4，步长为4，卷积核个数为48的二维卷积操作。

随后，每个Stage内部由Patch Merging模块（Stage1为Linear Embedding）、Swin Transformer模块组成。以Stage1、Stage2为例：

- Linear Embedding

  线性嵌入模块（Linear Embedding）将图像的通道数调整为嵌入长度C，调整后的图像尺寸为(H/4)×(W/4)×C。

- Swin Transformer

  Linear Embedding模块的输出序列长度往往较大，无法直接输入transformer。在本模型中，将输入张量划分为m×m大小的窗口，其中m为每个窗口内的patch数量，原文模型中默认为7。自注意力计算将在每个窗口内展开。为提取窗口间的信息，对窗口进行滑动，并再次计算自注意力。有关Swin Transformer模块的自注意力计算等实现细节见下文。

- Patch Merging

  下采样模块（Patch Merging）的作用是降低图片分辨率，扩大感受野，捕获多尺寸的图片信息，同时降低计算量，如图1(a)所示。类比于下采样在CNN模型中的实现，Swin Transformer模块的输出经过一次下采样，由(H/4)×(W/4)×C转换为(H/8)×(W/8)×2C，将临近的2个小patch合并成一个大patch，相当于进行了步长为2的二维卷积操作，后经过线性层（或1×1卷积层）调整通道数至2C，功能等价于Patch Partition模块与Linear Embedding模块的先后组合。此后再经过多个Swin Transformer模块、Patch Merging模块，进行多尺度特征的提取。

<center>
    <img src="https://github.com/microsoft/Swin-Transformer/blob/main/figures/teaser.png?raw=true" alt="image-20220819101847606" style="zoom:67%;" />
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 3px;">
    <center> 图1 Swin Transformer模型结构图 </center> </div>
</center>

### 1.3 Swin Transformer模块原理

Swin Transformer模块的细节实现如图1(c)所示，核心部分为多头自注意力的计算。如上所述，基于全尺寸图像的自注意力在面对密集计算型任务时具有较高的计算复杂度，因此本模型采用基于窗口的自注意力计算（W-MSA），输入张量先被分割成窗口，在窗口内的patch之间计算多头自注意力。相比于传统的自注意力计算，此处在计算q、k时额外增加了相对位置编码，以提供窗口内的位置信息。

Swin Transformer的亮点在于采用滑动窗口机制，提取窗口之间的信息，达到全局自注意力的效果。以4个4×4大小的窗口为例，在计算滑动窗口多头自注意力（SW-MSA）时，将窗口向右下移动2个patch，得到9个新窗口，其中上、下、左、右4个新窗口包含原窗口划分中2个窗口的图像信息，中间的新窗口包含原窗口划分中4个窗口的图像信息，从而实现窗口之间的通信。但随之而来的是窗口数量的成倍增加和不同的窗口大小，反而增大了计算难度。

为降低计算量，本模型对新窗口进行循环移位处理，组成新的窗口布局，将4×4大小的窗口数降低为4个，如图1(b)所示。此时除左上角的窗口外，其它3个窗口包含的部分patch原本并不属于同一区域，不应计算之间的自注意力，因此本模型创新性地提出了masked MSA机制，原理在自注意力计算结果上加上mask矩阵，目的是只取相同原窗口内的patch间的q、k计算结果，而不同原窗口内的patch间q、k计算结果添加原码后得到一个较大的负数，在随后的softmax层计算中，掩码部分输出将会是0，达到忽略其值的目的。

### 1.4 模型特点

- 基于滑动窗口的自注意力计算，交替使用W-MSA、SW-MSA，解决transformer应用于计算机视觉领域所产生的计算复杂度高的问题；
- 借鉴CNN架构，对图像进行降采样，捕获多尺寸层次的全局特征。

## 2 数据集简介

所使用的数据集：miniImageNet  
- 数据集大小：共100类，60000张图像，每类600张图像  
- 数据格式：JPGE格式，84*84彩色图像  
- 对数据集结构的处理要求：类别分布均衡，训练集 : 验证集 : 测试集 = 7 : 1 : 2  

miniImageNet数据集的原始结构如下：  
```text
└─ dataset
    ├─ images
        ├─ n0153282900000005.jpg
        ├─ n0153282900000006.jpg
        ├─ ...
    ├─ train.csv
    ├─ val.csv
    └─ test.csv
```
匹配图像与CSV文件后，数据集结构变为：  
```text
└─ dataset
    ├─ train
        ├─ 第1类
            └─ 600张图像 
        ├─ ...
        └─ 第64类
            └─ 600张图像 
    ├─ val
        ├─ 第65类
            └─ 600张图像 
        ├─ ...
        └─ 第80类
            └─ 600张图像 
    └─ test
        ├─ 第81类
            └─ 600张图像 
        ├─ ...
        └─ 第100类
            └─ 600张图像 
```
仍需进一步将数据集结构处理为：  
```text
└─ dataset
    ├─ train
        ├─ 第1类
            └─ 420张图像 
        ├─ ...
        └─ 第100类
            └─ 420张图像 
    ├─ val
        ├─ 第1类
            └─ 60张图像 
        ├─ ...
        └─ 第100类
            └─ 60张图像 
    └─ test
        ├─ 第1类
            └─ 120张图像 
        ├─ ...
        └─ 第100类
            └─ 120张图像 
```
处理后的数据集的下载方式：  
- 链接：https://pan.baidu.com/s/14d6sWeZiZS8Hzua0ohw4BA 提取码：xqnu  

请将数据集保存至路径“mindspore_swin_transformer/src/data”下，并逐层解压文件。

## 3 仓库说明

仓库结构为：  
```text
└─ swin_transformer
    ├─ src
        ├─ configs                // SwinTransformer的配置文件
            ├─ args.py
            └─ swin_tiny_patch4_window7_224.yaml
        └─ data
            ├─ augment            // 数据增强函数文件
                ├─ auto_augment.py
                ├─ mixup.py
                └─ random_erasing.py
            └─ imagenet           // miniImageNet数据集
                ├─ train
                ├─ val
                └─ test
    ├─ swin_transformer.ipynb     // 端到端可执行的Notebook文件
    └─ README.md
```

## 4 Notebook说明  

本案例的主体为端到端可执行的Notebook文件：**swin_transformer.ipynb**，该文件主要包括四个部分：  
### 4.1 SwinTransformer模型定义
模型类SwinTransformer的定义与继承关系如下：  
```text
SwinTransformer
    ├─ PatchEmbed
    └─ BasicLayer
        ├─ PatchMerging
        └─ SwinTransformerBlock
            ├─ WindowAttention
            ├─ RelativeBias
            ├─ DropPath1D
            ├─ Mlp
            ├─ Roll
            ├─ WindowPartitionConstruct
            └─ WindowReverseConstruct
```
### 4.2 miniImageNet数据集引入
- 从对应路径中读取训练/验证/测试数据
- 数据增强
### 4.3 训练
- 设置精度水平
- 定义Loss函数
- 定义单步训练
- 定义学习率、优化器和模型的获取函数
- 定义EvaluateCallBack，保存在验证集上指标最优的模型
- 载入数据、设置环境、初始化模型、训练
### 4.4 评估（测试）
- 载入在验证集上指标最优的模型
- 在测试集上测试

## 性能

miniImageNet上的SwinTransformer

| 属性 | 情况  |
| --- | --- |
| 模型 | SwinTransformer|
| 模型版本 | swin_tiny_patch4_window7_224 |
| 资源 | Gefore RTX 3090 * 1 |
| MindSpore版本 | 1.8.1 |
| 验证集 | miniImageNet Val，共6,000张图像 |
| 验证集分类准确率 | Top1-Acc: 55.32%, Top5-Acc: 81.05% |
| 测试集 | miniImageNet Test，共12,000张图像 |
| 测试集分类准确率 | Top1-Acc: 55.26%, Top5-Acc: 81.59% |