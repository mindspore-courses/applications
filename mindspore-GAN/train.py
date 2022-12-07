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
from src.TrainOneStep import *
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

# 选择执行模式为图模式；指定训练使用的平台为"GPU"，如需使用昇腾硬件可将其替换为"Ascend"
mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend")

# 获取处理后的数据集
dataset = create_dataset_train(batch_size=BATCH_SIZE, repeat_size=1, latent_size=latent_size)
# 获取数据集大小
iter_size = dataset.get_dataset_size()

# 利用随机种子创建一批隐码用来观察G
np.random.seed(2323)
test_noise = Tensor(np.random.normal(size=(25, latent_size)), dtype=mstype.float32)
random.shuffle(test_noise)

# 实例化生成器和判别器
netGenerator = Generator(latent_size)
netDiscriminator = Discriminator()
# 为生成器和判别器设置优化器
optimizerG = nn.Adam(netGenerator.trainable_params(), learning_rate=lr, beta1=b1, beta2=b2)
optimizerD = nn.Adam(netDiscriminator.trainable_params(), learning_rate=lr, beta1=b1, beta2=b2)
# 实例化WithLossCell
loss_G = GenWithLossCell(netGenerator, netDiscriminator, adversarial_loss)
loss_D = DisWithLossCell(netGenerator, netDiscriminator, adversarial_loss)
# 实例化TrainOneStepCell
GAN_train = TrainOneStepCell(loss_G, loss_D, optimizerG, optimizerD, gap)
# set train
netGenerator.set_train()
netDiscriminator.set_train()
# 储存loss和生成图片
G_losses, D_losses = [], []


for epoch in range(TOTLE_EPOCH):
    start = time.time()
    train_bar = tqdm(dataset_mnist, ncols=100, total=iter_size)
    for (iter, data) in enumerate(train_bar):
        image, latent_code = data
        image = (image - 127.5) / 127.5 # [0, 255] -> [-1, 1]
        image = ops.Reshape()(image, (image.shape[0], 1, image.shape[1], image.shape[2]))
        d_loss, g_loss = GAN_train(image, latent_code, iter)
        if iter % 100 == 0:
            print('[%3d/%d][%3d/%d]  Loss_D:%6.4f  Loss_G:%6.4f' % (epoch+1, TOTLE_EPOCH, iter+1, iter_size, d_loss.asnumpy(), g_loss.asnumpy()))
    D_losses.append(d_loss.asnumpy())
    G_losses.append(g_loss.asnumpy())
    end = time.time()
    print("time of epoch {} is {:.2f}s".format(epoch+1, end - start))

    # 每个epoch结束后，使用生成器生成一组图片
    netGenerator.set_train(False)
    gen_imgs = netGenerator(test_noise)
    netGenerator.set_train(True)
    save_imgs(gen_imgs.asnumpy(), epoch)

    # 保存网络模型参数为ckpt文件
    if(epoch % 10 == 0):
        save_checkpoint(netGenerator, checkpoints_path+"/Generator%d.ckpt" % (epoch))
        save_checkpoint(netDiscriminator, checkpoints_path+"/Discriminator%d.ckpt" % (epoch))
