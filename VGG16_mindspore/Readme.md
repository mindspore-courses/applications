# 项目介绍：

本程序以VGG16为例，用Mindspore框架进行VGG的数据集处理、网络搭建以及训练和测试。

VGGNet是牛津大学视觉几何组(Visual Geometry Group)提出的模型，该模型在2014ImageNet图像分类与定位挑战赛 ILSVRC-2014中取得在分类任务第二，定位任务第一的优异成绩。VGGNet突出的贡献是证明了很小的卷积，通过增加网络深度可以有效提高性能。VGG很好的继承了Alexnet的衣钵同时拥有着鲜明的特点。即网络层次较深。 VGGNet模型有A-E五种结构网络，深度分别为11,11,13,16,19。其中较为典型的网络结构主要有VGG16和VGG19。

##### 本程序使用的模型为VGG16,使用的数据集为CIFAR-10数据集。

###### 参考博客：https://blog.csdn.net/hgnuxc_1993/article/details/115956774

###### 参考博客：https://blog.csdn.net/weixin_43496706/article/details/10121098

# 数据集

### 这里使用了 ***download.download函数*** 来下载 ***CIFAR-10*** 数据集

### *`需要预先在控制台”pip install download“`*

CIFAR-10数据集由60000张32x32彩色图片组成，总共有10个类别，每类6000张图片。有50000个训练样本和10000个测试样本。10个类别包含飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车。
整个数据集被分为5个训练批次和1个测试批次，每一批10000张图片。测试批次包含10000张图片，是由每一类图片随机抽取出1000张组成的集合。剩下的50000张图片每一类的图片数量都是5000张，训练批次是由剩下的50000张图片打乱顺序，然后随机分成5份，所以可能某个训练批次中10个种类的图片数量不是对等的，会出现一个类的图片数量比另一类多的情况。

##### 会在同目录下创建一个名为datasets-cifar10-bin文件夹，然后自动将数据集下载至文件夹中。下载好之后结构如下：

datasets-cifar10-bin

└── cifar-10-batches-bin

​	├── data_batch_1.bin

​	├── data_batch_2.bin

​	├── data_batch_3.bin

​	├── data_batch_4.bin

​	├── data_batch_5.bin

​	├── test_batch.bin

​	├── readme.html

​	└── batches.meta.text

#### 如果不想下载的话，项目文件中也附带了下载好的数据集，保存在`./datasets-cifar10-bin`文件夹中，可以直接使用。

# 训练结果

### 在训练结束后，最优的参数会被保存在`./BestCheckpoint/vgg16-best.ckpt`中。

### 文件夹中的`./BestCheckpointSave/vgg16-best.ckpt`是我们训练了10轮之后的参数，准确率为55%，如果需要的话可以作为预训练参数使用，也可以用来直接测试效果。

# 附：

## 环境配置：

windows10

python 3.9

mindspore1.9.0 CPU版本

## 项目中的文件结构：

1. `./main.py`：项目代码，直接运行即可。
2. `./mindspore_VGG.ipynb`：附带网络模型介绍和代码逐段讲解的jupyter notebook文件，可以直接运行
3. `./BestCheckpointSave/vgg16-best.ckpt`：我们训练了40轮之后的参数，准确率为72.5%，如果需要的话可以作为预训练参数使用，也可以用来直接测试效果。
4. `./datasets-cifar10-bin`：训练和测试使用的数据集。
5. `./result.txt`：运行的完整输出

## 运行方式

直接运行main.py或mindspore_VGG.ipynb即可。

## VGG16网络的结构：

1、输入224x224x3的图片，经64个3x3的卷积核作两次卷积+ReLU，卷积后的尺寸变为224x224x64

2、作max pooling（最大化池化），池化单元尺寸为2x2（效果为图像尺寸减半），池化后的尺寸变为112x112x64

3、经128个3x3的卷积核作两次卷积+ReLU，尺寸变为112x112x128

4、作2x2的max pooling池化，尺寸变为56x56x128

5、经256个3x3的卷积核作三次卷积+ReLU，尺寸变为56x56x256

6、作2x2的max pooling池化，尺寸变为28x28x256

7、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为28x28x512

8、作2x2的max pooling池化，尺寸变为14x14x512

9、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为14x14x512

10、作2x2的max pooling池化，尺寸变为7x7x512

11、与两层1x1x4096，一层1x1x1000进行全连接+ReLU（共三层）

12、通过softmax输出1000个预测结果(最终会取可能性最大的那个预测结果作为最终预测输出)







