# Copyright 2022 Huawei Technologies Co., Ltd
# reference by: https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/net/generator.py
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
"""Generator network."""

import mindspore.nn as nn
from mindspore.common.initializer import Normal

from .conv2d_block import ConvBlock
from .inverted_residual_block import InvertedResBlock
from .upsample import UpSample


class Generator(nn.Cell):
    """
    Generator network.

    Args:
        channels (int): Base channel number per layer.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output from the generator network.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> generator = Generator()
    """

    def __init__(self, channels=32):
        super(Generator, self).__init__()
        has_bias = False

        self.generator = nn.SequentialCell()
        self.generator.append(ConvBlock(3, channels, kernel_size=7))
        self.generator.append(ConvBlock(channels, channels * 2, stride=2))
        self.generator.append(ConvBlock(channels * 2, channels * 4, stride=2))
        self.generator.append(ConvBlock(channels * 4, channels * 4))
        self.generator.append(ConvBlock(channels * 4, channels * 4))

        self.generator.append(InvertedResBlock(channels * 4, channels * 8))
        self.generator.append(InvertedResBlock(channels * 8, channels * 8))
        self.generator.append(InvertedResBlock(channels * 8, channels * 8))
        self.generator.append(InvertedResBlock(channels * 8, channels * 8))
        self.generator.append(ConvBlock(channels * 8, channels * 4))

        self.generator.append(UpSample(channels * 4, channels * 4))
        self.generator.append(ConvBlock(channels * 4, channels * 4))

        self.generator.append(UpSample(channels * 4, channels * 2))
        self.generator.append(ConvBlock(channels * 2, channels * 2))
        self.generator.append(ConvBlock(channels * 2, channels, kernel_size=7))
        self.generator.append(
            nn.Conv2d(channels, 3, kernel_size=1, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=has_bias))
        self.generator.append(nn.Tanh())

    def construct(self, x):
        """ build network """
        out = self.generator(x)
        return out
