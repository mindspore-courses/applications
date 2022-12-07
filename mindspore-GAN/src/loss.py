from mindspore import nn
import mindspore.ops.functional as F

'''定义损失函数'''
adversarial_loss = nn.BCELoss(reduction='mean')

'''连接生成器和损失'''
class GenWithLossCell(nn.Cell):
    def __init__(self, netG, netD, loss_fn, auto_prefix=True):
        super(GenWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = loss_fn
    '''构建生成器损失计算结构'''
    def construct(self, latent_code2):
        fake_data = self.netG(latent_code2)
        fake_out = self.netD(fake_data)
        loss_G = self.loss_fn(fake_out, F.ones_like(fake_out))
        return loss_G

'''连接判别器和损失'''
class DisWithLossCell(nn.Cell):
    def __init__(self, netG, netD, loss_fn, auto_prefix=True):
        super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = loss_fn
    '''构建判别器损失计算结构'''
    def construct(self, real_data, latent_code1):
        fake_data = self.netG(latent_code1)
        real_out = self.netD(real_data)
        real_loss = self.loss_fn(real_out, F.ones_like(real_out))
        fake_out = self.netD(fake_data)
        fake_loss = self.loss_fn(fake_out, F.zeros_like(fake_out))
        loss_D = real_loss + fake_loss
        return loss_D