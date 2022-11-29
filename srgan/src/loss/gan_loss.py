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
""""GAN Loss"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from src.loss.mean_shift import MeanShift

__all__ = ["DiscriminatorLoss", "GeneratorLoss"]

class DiscriminatorLoss(nn.Cell):
    """
    Loss for discriminator.

    Args:
        discriminator (nn.Cell): Discriminator network.
        generator (nn.Cell): SRResnet.

    Inputs:
        - **hr_img** (Tensor) - The high-resolution image.
          The input shape must be (batchsize, num_channels, height, width).
        - **lr_img** (Tensor) - The low-resolution image.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **d_loss** (Tensor) -  Binary cross-entropy loss.
          The output has the shape (batchsize, loss_value).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from model.generator import get_generator
        >>> from model.discriminator import get_discriminator
        >>> generator = get_generator(4, 0.02)
        >>> discriminator = get_discriminator(96, 0.02)
        >>> discriminator_loss = DiscriminatorLoss(discriminator, generator)
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> lr_img = Tensor(np.zeros([16, 3, 24, 24]),mstype.float32)
        >>> loss_value = discriminator_loss(hr_img, lr_img)
        >>> print(loss_value)
        [[1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]
        [1.3862944]]
    """
    def __init__(self, discriminator, generator):
        super(DiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.adversarial_criterion = nn.BCELoss()
        ones = ops.Ones()
        zeros = ops.Zeros()
        self.real_lable = ones((16, 1), mstype.float32)
        self.fake_lable = zeros((16, 1), mstype.float32)

    def construct(self, hr_img, lr_img):
        """dloss"""
        # Generating fake high resolution images from real low resolution images.
        sr = self.generator(lr_img)
        # Let the discriminator realize that the sample is real.
        real_output = self.discriminator(hr_img)
        d_loss_real = self.adversarial_criterion(real_output, self.real_lable)
        # Let the discriminator realize that the sample is false.
        fake_output = self.discriminator(sr)
        d_loss_fake = self.adversarial_criterion(fake_output, self.fake_lable)
        d_loss = d_loss_fake+d_loss_real
        return  d_loss

class GeneratorLoss(nn.Cell):
    """
    Loss for generator.

    Args:
        discriminator (nn.Cell): Discriminator network.
        generator (nn.Cell): SRResnet.
        vgg_ckpt (str): The path of vgg19 checkpoint file.

    Inputs:
        - **hr_img** (Tensor) - The high-resolution image.
          The input shape must be (batchsize, num_channels, height, width).
        - **lr_img** (Tensor) - The low-resolution image.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **g_loss** (Tensor) -  Generator loss, compose of perception loss, adversarial loss and l2 loss.
          The output has the shape (batchsize, loss_value).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from model.generator import get_generator
        >>> from model.discriminator import get_discriminator
        >>> from vgg19.define import vgg19
        >>> from loss.mean_shift import MeanShift
        >>> generator = get_generator(4, 0.02)
        >>> discriminator = get_discriminator(96, 0.02)
        >>> generator_loss = GeneratorLoss(discriminator, generator, './vgg19/vgg19.ckpt')
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> lr_img = Tensor(np.zeros([16, 3, 24, 24]),mstype.float32)
        >>> loss_value = generator_loss(hr_img, lr_img)
        >>> print(loss_value)
        [[0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]]
    """
    def __init__(self, discriminator, generator, vgg):
        super(GeneratorLoss, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.mse_loss = nn.MSELoss()
        self.adversarial_criterion = nn.BCELoss()
        ones = ops.Ones()
        self.real_lable = ones((16, 1), mstype.float32)
        self.meanshif = MeanShift()
        self.vgg = vgg
        for p in self.meanshif.get_parameters():
            p.requires_grad = False

    def construct(self, hr_img, lr_img):
        """gloss"""
        # L2loss
        sr = self.generator(lr_img)
        l2_loss = self.mse_loss(sr, hr_img)

        # adversarialloss
        fake_output = self.discriminator(sr)
        adversarial_loss = self.adversarial_criterion(fake_output, self.real_lable)

        # vggloss
        hr_img = (hr_img+1.0)/2.0
        sr = (sr+1.0)/2.0
        hr_img = self.meanshif(hr_img)
        sr = self.meanshif(sr)
        hr_feat = self.vgg(hr_img)
        sr_feat = self.vgg(sr)
        percep_loss = self.mse_loss(hr_feat, sr_feat)

        g_loss = 0.006 * percep_loss + 1e-3 * adversarial_loss + l2_loss
        return  g_loss
