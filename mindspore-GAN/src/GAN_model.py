from mindspore import nn
from src.configs import *

class Generator(nn.Cell):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.SequentialCell(
            # [N, 100] -> [N, 128]
            nn.Dense(latent_dim, 128), #输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维
            nn.ReLU(),
            # [N, 128] -> [N, 256]
            nn.Dense(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # [N, 256] -> [N, 512]
            nn.Dense(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # [N, 512] -> [N, 1024]
            nn.Dense(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # [N, 1024] -> [N, 784]
            nn.Dense(1024, img_size * img_size),
            nn.Tanh(),  # 经过Tanh激活函数是希望生成的假的图片数据分布能够在[-1.0, 1.0]之间
        )

    def construct(self, x):
        x = self.model(x)
        return x.view(x.shape[0], channels, img_size, img_size)

class Discriminator(nn.Cell):
    def __init__(self, auto_prefix=True):
        super().__init__(auto_prefix=auto_prefix)
        self.model = nn.SequentialCell(
            # [N, 784] -> [N, 512]
            nn.Dense(img_size * img_size, 512),  #输入特征数为784，输出为512
            nn.LeakyReLU(),  #默认斜率为0.2的非线性映射激活函数
            # [N, 512] -> [N, 256]
            nn.Dense(512, 256),  #进行一个线性映射
            nn.LeakyReLU(),
            # [N, 256] -> [N, 1]
            nn.Dense(256, 1),
            nn.Sigmoid(),  #二分类激活函数，将实数映射到[0,1]
        )
    
    def construct(self, x):
        x_flat = x.view(x.shape[0], -1)
        return self.model(x_flat)
    