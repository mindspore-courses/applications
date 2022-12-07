import numpy as np
from mindspore import nn
from mindspore.communication import init
from src.loss import *
from src.GAN_model import *

'''TrainOneStepCell'''
class TrainOneStepCell(nn.Cell):
    def __init__(
        self,
        netG: GenWithLossCell,
        netD: DisWithLossCell,
        optimizerG1: nn.Optimizer,
        optimizerD1: nn.Optimizer,
        gap: int,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netG.set_grad()  # 生成需要计算梯度的反向网络
        self.netG.add_flags(defer_inline=True)

        self.netD = netD
        self.netD.set_grad()
        self.netD.add_flags(defer_inline=True)

        self.weights_G = optimizerG1.parameters
        self.optimizerG1 = optimizerG1
        self.weights_D = optimizerD1.parameters
        self.optimizerD1 = optimizerD1

        self.grad = C.GradOperation(get_by_list=True, sens_param=True) # 灵敏度：返回的梯度乘以灵敏度

        self.sens = sens

        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity

        self.gap = gap

    # train discriminator
    def trainD(self, real_data, latent_code, loss, loss_net, grad, optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = grad(loss_net, weights)(real_data, latent_code, sens)
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss

    # train generator
    def trainG(self, latent_code4, loss, loss_net, grad, optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = grad(loss_net, weights)(latent_code4, sens) # 先得到gradient function, 再带入参数; sens是grad_wrt_output
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss

    def construct(self, real_data, latent_code5, now_iter):
        d_loss, g_loss = -1, -1
        if now_iter % self.gap == 0:
            loss_D = self.netD(real_data, latent_code5)
            d_loss = self.trainD(real_data, latent_code5, loss_D, self.netD, self.grad, self.optimizerD1, self.weights_D, self.grad_reducer_D)
        
        loss_G = self.netG(latent_code5)
        g_loss = self.trainG(latent_code5, loss_G, self.netG, self.grad, self.optimizerG1, self.weights_G, self.grad_reducer_G)
        return d_loss, g_loss