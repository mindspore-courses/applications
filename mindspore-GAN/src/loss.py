from mindspore import nn
import mindspore.ops.functional as F

# 定义损失函数
adversarial_loss = nn.BCELoss(reduction='mean')

class GenWithLossCell(nn.Cell):
    def __init__(self, netG, netD, loss_fn):
        super(GenWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.loss_fn = loss_fn
    '''构建生成器损失计算结构'''
    def construct(self, latent):
        fake_img = self.netG(latent)
        fake_out = self.netD(fake_img)
        loss_G = self.loss_fn(fake_out, F.ones_like(fake_out))
        return loss_G

class DisWithLossCell(nn.Cell):
    def __init__(self, netG, netD, loss_fn):
        super(DisWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.loss_fn = loss_fn
    '''构建判别器损失计算结构'''
    def construct(self, real_data, latent):
        fake_img = self.netG(latent)
        fake_out = self.netD(fake_img)
        real_out = self.netD(real_data)

        d_fake_loss = self.loss_fn(fake_out, F.zeros_like(fake_out))
        d_real_loss = self.loss_fn(real_out, F.ones_like(real_out))

        loss_D = d_real_loss + d_fake_loss
        return loss_D