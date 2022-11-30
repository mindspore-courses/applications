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
"""Loss function"""

import random

import mindspore as ms
from mindspore import nn, Tensor, ops


class StyleGANLoss(nn.Cell):
    """
    StyleGANLoss.

    Args:
        g_mapping (nn.Cell): Generator mapping network.
        g_synthesis (nn.Cell): Generator synthesis network.
        discriminator (nn.Cell): Discriminator.
        style_mixing_prob (float): Style mixing probability. Default=0.9.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = StyleGANLoss(g_mapping, g_synthesis, discriminator)
    """
    def __init__(self, g_mapping, g_synthesis, discriminator, style_mixing_prob=0.9):
        super().__init__()
        self.g_mapping = g_mapping
        self.g_synthesis = g_synthesis
        self.discriminator = discriminator
        self.style_mixing_prob = style_mixing_prob

    def run_g(self, g_z, pose, g_c):
        """
        Run the generator.

        Args:
            g_z (Tensor): Latent tensor for the generator.
            pose (Tensor): Pose tensor.
            g_c (Tensor): Label tensor for the generator.

        Returns:
            Tensor, output image.

        Examples:
            >>> gen_img, gen_ws, pose_enc = run_g(gen_z, pose, gen_c)
        """
        ws = self.g_mapping.construct(g_z, g_c)
        if self.style_mixing_prob > 0:
            stdnormal = ops.StandardNormal()
            shape_1 = ws.shape[1]
            cutoff = random.randint(1, shape_1)
            if random.random() >= 0.9:
                cutoff = ws.shape[1]
            ws[:, cutoff:] = self.g_mapping.construct(stdnormal(g_z.shape), g_c, skip_w_avg_update=True)[:, cutoff:]
        img = self.g_synthesis.construct(ws, pose)
        return img

    def run_d(self, img, d_c):
        """
        Run the discriminator.

        Args:
            img (Tensor): Input image tensor.
            d_c (Tensor): Label tensor for the discriminator.

        Returns:
            Tensor, output logits tensor.

        Examples:
            >>> gen_logits = run_d(gen_img, gen_c)
        """
        logits = self.discriminator.construct(img, d_c)
        return logits

    def accumulate_gradients(self, do_gmain, do_dmain, real_img, pose, real_c, gen_z, gen_c, gain):
        """
        Accumulate gradients.

        Args:
            do_gmain (bool): Conduct the generator.
            do_dmain (bool): Conduct the discriminator.
            real_img (Tensor): Real images tensor.
            pose (Tensor): Poses tensor.
            real_c (Tensor): Real labels tensor.
            gen_z (Tensor): Latent tensor.
            gen_c (Tensor): Latent labels tensor.
            gain (int): Loss gain.

        Returns:
            Tensor, total loss.

        Examples:
            >>> loss = accumulate_gradients(do_gmain, do_dmain, real_img, pose, real_c, gen_z, gen_c, gain)
        """
        mul = ops.Mul()
        softplus = ops.Softplus()

        if do_gmain:
            gen_img = self.run_g(gen_z, pose, gen_c)
            gen_logits = self.run_d(gen_img, gen_c)
            loss_gmain = softplus(-gen_logits)
            loss = mul(loss_gmain.mean(), gain)
            return loss

        loss_dgen = 0
        if do_dmain:
            gen_img = self.run_g(gen_z, pose, gen_c)
            gen_logits = self.run_d(gen_img, gen_c)
            loss_dgen = softplus(gen_logits)
            loss1 = mul(loss_dgen.mean(), gain)
        else:
            loss1 = loss_dgen

        if do_dmain:
            real_img_tmp = real_img
            real_logits = self.run_d(real_img_tmp, real_c)
            loss_dreal = 0
            if do_dmain:
                loss_dreal = softplus(-real_logits)
            loss2 = mul((real_logits * 0 + loss_dreal).mean(), gain)
            d_total_loss = loss1 + loss2
            return d_total_loss
        loss_zero = Tensor(0, ms.float32)
        return loss_zero


class CustomWithLossCell(nn.Cell):
    """
    CustomWithLossCell for training.

    Args:
        g_mapping (nn.Cell): Generator mapping network.
        g_synthesis (nn.Cell): Generator synthesis network.
        discriminator (nn.Cell): Discriminator.
        style_gan_loss (nn.Cell): StyleGAN loss.

    Inputs:
        - **do_gmain** (bool) - Conduct the generator.
        - **do_dmain** (bool) - Conduct the discriminator.
        - **real_img** (Tensor) - Real images.
        - **pose** (Tensor) - Poses.
        - **real_c** (Tensor) - Real labels.
        - **gen_z** (Tensor) - Latent tensor.
        - **gen_c** (Tensor) - Latent labels.
        - **gain** (Tensor) - Loss gain.

    Outputs:
        Tensor, a float tensor representing the total loss.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> cal_loss = CustomWithLossCell(generator.mapping, generator.synthesis, discriminator, StyleGANLoss)
        >>> network = nn.TrainOneStepCell(cal_loss, opt)
        >>> network(do_gmain, do_dmain, real_img, pose, real_c, gen_z, gen_c, gain)
    """
    def __init__(self, g_mapping, g_synthesis, discriminator, style_gan_loss):
        super(CustomWithLossCell, self).__init__()
        self.g_mapping = g_mapping
        self.g_synthesis = g_synthesis
        self.discriminator = discriminator
        self.style_gan_loss = style_gan_loss(self.g_mapping, self.g_synthesis, self.discriminator)

    def construct(self, do_gmain, do_dmain, real_img, pose, real_c, gen_z, gen_c, gain):
        """CustomWithLossCell construct"""
        loss = self.style_gan_loss.accumulate_gradients(do_gmain, do_dmain, real_img, pose, real_c, gen_z, gen_c, gain)
        return loss
