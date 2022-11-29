# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cycle GAN losses"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class BCEWithLogits(nn.Cell):
    """
    BCEWithLogits creates a criterion to measure the Binary Cross Entropy between the true labels and
    predicted labels with sigmoid logits.

    Args:
        reduction (str): Specifies the reduction to be applied to the output.
        The optional values are 'none', 'mean', 'sum'. Default: 'mean'.

    Outputs:
        Tensor or Scalar, if 'reduction' is 'none', then output is a tensor and has the same shape as 'inputs'.
        Otherwise, the output is a scalar.
    """

    def __init__(self, reduction='mean'):
        super(BCEWithLogits, self).__init__()
        if not reduction:
            reduction = 'none'
        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.reduce = False
        if reduction == 'sum':
            self.reduce_mode = ops.ReduceSum()
            self.reduce = True
        elif reduction == 'mean':
            self.reduce_mode = ops.ReduceMean()
            self.reduce = True

    def construct(self, predict, target):
        loss = self.loss(predict, target)
        if self.reduce:
            loss = self.reduce_mode(loss)
        return loss


class GANLoss(nn.Cell):
    """
    The GANLoss class abstracts away the need to create the target label tensor that has the same size as the input.

    Args:
        mode (str): The type of GAN objective. It currently supports 'vanilla', 'lsgan'. Default: 'lsgan'.
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'mean'.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `inputs`.
        Otherwise, the output is a scalar.
    """

    def __init__(self, mode="lsgan", reduction='mean'):
        super(GANLoss, self).__init__()
        self.loss = None
        self.ones = ops.OnesLike()
        if mode == "lsgan":
            self.loss = nn.MSELoss(reduction)
        elif mode == "vanilla":
            self.loss = BCEWithLogits(reduction)
        else:
            raise NotImplementedError(f'GANLoss {mode} not recognized, we support lsgan and vanilla.')

    def construct(self, predict, target):
        target = ops.cast(target, ops.dtype(predict))
        target = self.ones(predict) * target
        loss = self.loss(predict, target)
        return loss


class GeneratorLoss(nn.Cell):
    """
    Cycle GAN generator loss.

    Args:
        args (class): Option class.
        generator (Cell): Generator of CycleGAN.
        d_a (Cell): The discriminator network of domain A to domain B.
        d_b (Cell): The discriminator network of domain B to domain A.

    Outputs:
        Tuple Tensor, the losses of generator.
    """

    def __init__(self, args, generator, d_a, d_b):
        super(GeneratorLoss, self).__init__()
        self.lambda_a = args.lambda_a
        self.lambda_b = args.lambda_b
        self.lambda_idt = args.lambda_idt
        self.use_identity = args.lambda_idt > 0
        self.dis_loss = GANLoss(args.gan_mode)
        self.rec_loss = nn.L1Loss("mean")
        self.generator = generator
        self.d_a = d_a
        self.d_b = d_b
        self.true = Tensor(True, mstype.bool_)

    def construct(self, img_a, img_b):
        """
        If use_identity, identity loss will be used.

        Args:
            img_a(numpy array / Tensor): image of domain A.
            img_b(numpy array / Tensor): image of domain B.
        """

        fake_a, fake_b, rec_a, rec_b, identity_a, identity_b = self.generator(img_a, img_b)
        loss_g_a = self.dis_loss(self.d_b(fake_b), self.true)
        loss_g_b = self.dis_loss(self.d_a(fake_a), self.true)
        loss_c_a = self.rec_loss(rec_a, img_a) * self.lambda_a
        loss_c_b = self.rec_loss(rec_b, img_b) * self.lambda_b
        if self.use_identity:
            loss_idt_a = self.rec_loss(identity_a, img_a) * self.lambda_a * self.lambda_idt
            loss_idt_b = self.rec_loss(identity_b, img_b) * self.lambda_b * self.lambda_idt
        else:
            loss_idt_a = 0
            loss_idt_b = 0
        loss_g = loss_g_a + loss_g_b + loss_c_a + loss_c_b + loss_idt_a + loss_idt_b
        return (fake_a, fake_b, loss_g, loss_g_a, loss_g_b, loss_c_a, loss_c_b, loss_idt_a, loss_idt_b)


class DiscriminatorLoss(nn.Cell):
    """
    Cycle GAN discriminator loss.

    Args:
        args (class): option class.
        d_a (Cell): The discriminator network of domain A to domain B.
        d_b (Cell): The discriminator network of domain B to domain A.

    Outputs:
        Tuple Tensor, the loss of discriminator.
    """

    def __init__(self, args, d_a, d_b):
        super(DiscriminatorLoss, self).__init__()
        self.d_a = d_a
        self.d_b = d_b
        self.false = Tensor(False, mstype.bool_)
        self.true = Tensor(True, mstype.bool_)
        self.dis_loss = GANLoss(args.gan_mode)
        self.rec_loss = nn.L1Loss("mean")

    def construct(self, img_a, img_b, fake_a, fake_b):
        d_fake_a = self.d_a(fake_a)
        d_img_a = self.d_a(img_a)
        d_fake_b = self.d_b(fake_b)
        d_img_b = self.d_b(img_b)
        loss_d_a = self.dis_loss(d_fake_a, self.false) + self.dis_loss(d_img_a, self.true)
        loss_d_b = self.dis_loss(d_fake_b, self.false) + self.dis_loss(d_img_b, self.true)
        loss_d = (loss_d_a + loss_d_b) * 0.5
        return loss_d
