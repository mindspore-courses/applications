import numpy as np
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C
from src.configs import *

# 重写reshape方法
class Reshape(nn.Cell):
    def __init__(self, shape, auto_prefix=True):
        super().__init__(auto_prefix=auto_prefix)
        self.shape = shape
        self.reshape = P.Reshape()

    def construct(self, x):
        return self.reshape(x, self.shape)

class Generator(nn.Cell):
    def __init__(self, latent_size, auto_prefix=True):
        super(Generator, self).__init__(auto_prefix=auto_prefix)
        self.model = nn.SequentialCell()
        # [N, 100] -> [N, 128]
        self.model.append(nn.Dense(latent_size, 128))  #输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维
        self.model.append(nn.ReLU())
        # [N, 128] -> [N, 256]
        self.model.append(nn.Dense(128, 256))
        self.model.append(nn.BatchNorm1d(256))
        self.model.append(nn.ReLU())
        # [N, 256] -> [N, 512]
        self.model.append(nn.Dense(256, 512))
        self.model.append(nn.BatchNorm1d(512))
        self.model.append(nn.ReLU())
        # [N, 512] -> [N, 1024]
        self.model.append(nn.Dense(512, 1024))
        self.model.append(nn.BatchNorm1d(1024))
        self.model.append(nn.ReLU())
        # [N, 1024] -> [N, 784]
        self.model.append(nn.Dense(1024, img_size * img_size))  # 经过线性变换将其变成784维
        self.model.append(nn.Tanh())  # 经过Tanh激活函数是希望生成的假的图片数据分布能够在-1～1之间

    def construct(self, x):
        img = self.model(x)
        return Reshape((-1, 1, 28, 28))(img)


class Discriminator(nn.Cell):
    def __init__(self, auto_prefix=True):
        super().__init__(auto_prefix=auto_prefix)
        self.model = nn.SequentialCell()
        # [N, 784] -> [N, 512]
        self.model.append(nn.Dense(img_size * img_size, 512))  #输入特征数为784，输出为512
        self.model.append(nn.LeakyReLU())  #默认斜率为0.2的非线性映射激活函数
        # [N, 512] -> [N, 256]
        self.model.append(nn.Dense(512, 256))  #进行一个线性映射
        self.model.append(nn.LeakyReLU())
        # [N, 256] -> [N, 1]
        self.model.append(nn.Dense(256, 1))
        self.model.append(nn.Sigmoid())  #二分类激活函数，将实数映射到[0,1]
    
    def construct(self, x):
        x_flat = Reshape((-1, img_size * img_size))(x)
        return self.model(x_flat)