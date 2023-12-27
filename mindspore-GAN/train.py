import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import mindspore
from tqdm import tqdm
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, context, save_checkpoint
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore.communication import init
from src.loss import *
from data_loader import *
from src.configs import *
from src.GAN_model import *
from src.utils import *
import mindspore
# 设置参数保存路径
checkpoints_path = "./result/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

# 设置中间过程生成图片保存路径
image_path = "./result/images"
os.makedirs(image_path, exist_ok=True)
def save_imgs(gen_imgs1, idx): # 保存生成的test图像
    for i3 in range(gen_imgs1.shape[0]):
        plt.subplot(5, 5, i3 + 1)
        plt.imshow(gen_imgs1[i3, 0, :, :]/2+0.5, cmap="gray")
        plt.axis("off")
    plt.savefig(image_path+"/{}.png".format(idx))

# Mindspore2.0默认执行模式为动态图模式(PYNATIVE_MODE)，指定训练使用的平台为"GPU"，如需使用昇腾硬件可将其替换为"Ascend"
mindspore.set_context(device_target="GPU")

# 获取处理后的数据集
train_dataset = get_train_dataset(batch_size=BATCH_SIZE)

# 利用随机种子创建一批隐码用来观察G
np.random.seed(2323)
test_noise = Tensor(np.random.normal(size=(25, LATENT_DIM)), dtype=mstype.float32)
random.shuffle(test_noise)

# 实例化生成器和判别器
netGenerator = Generator(LATENT_DIM)
netDiscriminator = Discriminator()
# 为生成器和判别器设置优化器
optimizerG = nn.Adam(netGenerator.trainable_params(), learning_rate=lr, beta1=b1, beta2=b2)
optimizerD = nn.Adam(netDiscriminator.trainable_params(), learning_rate=lr, beta1=b1, beta2=b2)
# 实例化WithLossCell
g_loss_fn = GenWithLossCell(netGenerator, netDiscriminator, adversarial_loss)
d_loss_fn = DisWithLossCell(netGenerator, netDiscriminator, adversarial_loss)
# set train
netGenerator.set_train()
netDiscriminator.set_train()

# 定义单次训练过程的所需要的函数以及单次训练更新参数的流程
def generator_forward_fn(noise):
    out = netGenerator(noise)
    loss = g_loss_fn(noise)
    return loss, out

def discriminator_forward_fn(real_img, noise):
    out = netDiscriminator(netGenerator(noise))
    loss = d_loss_fn(real_img, noise)
    return loss, out

generator_grad_fn = mindspore.value_and_grad(generator_forward_fn, None, optimizerG.parameters, has_aux=True)
discriminator_grad_fn = mindspore.value_and_grad(discriminator_forward_fn, None, optimizerD.parameters, has_aux=True)

def train_step(real_img, noise):
    (g_loss, _), g_grads = generator_grad_fn(noise)
    optimizerG(g_grads)
    (d_loss, _), d_grads = discriminator_grad_fn(real_img, noise)
    optimizerD(d_grads)
    return g_loss, d_loss

# 储存loss和生成图片
G_losses, D_losses = [], []

for epoch in range(TRAIN_EPOCH):
    start = time.time()
    iter_size = train_dataset.get_dataset_size()
    train_bar = tqdm(train_dataset, ncols=100, total=iter_size)
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    for (i, batch) in enumerate(train_bar):
        batch_imgs, _ = batch
        batch_size = batch_imgs.shape[0]
        batch_noise = Tensor(np.random.normal(size=(batch_size, LATENT_DIM)), dtype=mstype.float32)
        random.shuffle(batch_noise)
        g_loss, d_loss = train_step(batch_imgs, batch_noise)
        if iter % 100 == 0:
            print('[%3d/%d][%3d/%d]  Loss_D:%6.4f  Loss_G:%6.4f' % (epoch+1, TRAIN_EPOCH, i+1, iter_size, d_loss.asnumpy(), g_loss.asnumpy()))
        g_loss_epoch += g_loss.asnumpy()
        d_loss_epoch += d_loss.asnumpy()
    D_losses.append(d_loss.asnumpy()/iter_size)
    G_losses.append(g_loss.asnumpy()/iter_size)
    end = time.time()
    print("time of epoch {} is {:.2f}s".format(epoch+1, end - start))

    # 每个epoch结束后，使用生成器生成一组图片
    netGenerator.set_train(False)
    gen_imgs = netGenerator(test_noise)
    netGenerator.set_train(True)
    save_imgs(gen_imgs.asnumpy(), epoch)

    # 保存网络模型参数为ckpt文件
    if(epoch % 5 == 0):
        save_checkpoint(netGenerator, checkpoints_path+"/Generator%d.ckpt" % (epoch))
        save_checkpoint(netDiscriminator, checkpoints_path+"/Discriminator%d.ckpt" % (epoch))