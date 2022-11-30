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
"""Define with losses."""

import mindspore
from mindspore import nn, ops

from .cra_utils.utils import gan_wgan_loss, random_interpolates, GradientsPenalty


class GenWithLossCell(nn.Cell):
    """
    Build the generator loss.

    Args:
        net_g(cell): generator network.
        net_d(cell): discriminator network.
        args(class): option class.
        auto_prefix(bool): whether to automatically generate namespace for cell and its subcells.
            If set to True, the network parameter name will be prefixed, otherwise it will not.

    Return:
        loss_g: the loss of generator.
    """

    def __init__(self, net_g, net_d, args, auto_prefix=True):
        super(GenWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.net_g = net_g
        self.net_d = net_d
        self.gan_wgan_loss = gan_wgan_loss
        self.coarse_alpha = args.coarse_alpha
        self.gan_with_mask = args.gan_with_mask
        self.gan_loss_alpha = args.gan_loss_alpha
        self.in_hole_alpha = args.in_hole_alpha
        self.context_alpha = args.context_alpha
        self.train_batchsize = args.train_batchsize
        self.mean = ops.ReduceMean(False)
        self.abs = ops.Abs()
        self.concat_0 = ops.Concat(0)
        self.concat_1 = ops.Concat(1)
        self.split = ops.Split(0, 2)
        self.tile = ops.Tile()

    def construct(self, real, x, mask):
        """compute generator losses"""

        x1, x2, _ = self.net_g(x, mask)
        fake = x2
        losses = {}
        fake_patched = fake * mask + real * (1 - mask)
        fake_patched = fake_patched.astype(mindspore.float32)
        losses['in_hole_loss'] = self.coarse_alpha * self.mean(self.abs(real - x1) * mask)
        losses['in_hole_loss'] = losses['in_hole_loss'] + self.mean(self.abs(real - x2) * mask)
        losses['context_loss'] = self.coarse_alpha * self.mean(self.abs(real - x1) * (1 - mask))
        losses['context_loss'] = losses['context_loss'] + self.mean(self.abs(real - x2) * (1 - mask))
        losses['context_loss'] = losses['context_loss'] / self.mean(1 - mask)
        real_fake = self.concat_0((real, fake_patched))
        if self.gan_with_mask:
            real_fake = self.concat_1((real_fake, self.tile(mask, (self.train_batchsize * 2, 1, 1, 1))))
        d_real_fake = self.net_d(real_fake)
        d_real, d_fake = self.split(d_real_fake)
        g_loss, _ = self.gan_wgan_loss(d_real, d_fake)
        losses['adv_gloss'] = g_loss
        losses['g_loss'] = self.gan_loss_alpha * losses['adv_gloss']
        losses['g_loss'] = losses['g_loss'] + self.in_hole_alpha * losses['in_hole_loss']
        losses['g_loss'] = losses['g_loss'] + self.context_alpha * losses['context_loss']
        loss_g = losses['g_loss']
        return loss_g


class DisWithLossCell(nn.Cell):
    """
    Build the discriminator loss.

    Args:
        net_g(cell): generator network.
        net_d(cell): discriminator network.
        args(class): option class.
        auto_prefix(bool): whether to automatically generate namespace for cell and its subcells.
            If set to True, the network parameter name will be prefixed, otherwise it will not.

    Return:
        loss_d: the loss of discriminator.
    """

    def __init__(self, net_g, net_d, args, auto_prefix=True):
        super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.net_g = net_g
        self.net_d = net_d
        self.gan_wgan_loss = gan_wgan_loss
        self.random_interpolates = random_interpolates
        self.gradients_penalty = GradientsPenalty(self.net_d)
        self.gan_with_mask = args.gan_with_mask
        self.wgan_gp_lambda = args.wgan_gp_lambda
        self.train_batchsize = args.train_batchsize
        self.concat_0 = ops.Concat(0)
        self.concat_1 = ops.Concat(1)
        self.split = ops.Split(0, 2)

    def construct(self, real, x, mask):
        """compute discriminator losses"""

        _, x2, _ = self.net_g(x, mask)
        fake = x2
        losses = {}
        fake_patched = fake * mask + real * (1 - mask)
        fake_patched = fake_patched.astype(mindspore.float32)
        real_fake = self.concat_0((real, fake_patched))
        if self.gan_with_mask:
            real_fake = self.concat_1((real_fake, ops.Tile()(mask, (self.train_batchsize * 2, 1, 1, 1))))
        d_real_fake = self.net_d(real_fake)
        d_real, d_fake = self.split(d_real_fake)
        _, d_loss = self.gan_wgan_loss(d_real, d_fake)
        losses['adv_dloss'] = d_loss
        interps = self.random_interpolates(real, fake_patched)
        gp_loss = self.gradients_penalty(interps)
        losses['gp_loss'] = self.wgan_gp_lambda * gp_loss
        losses['d_loss'] = losses['adv_dloss'] + losses['gp_loss']
        loss_d = losses['d_loss']
        return loss_d
