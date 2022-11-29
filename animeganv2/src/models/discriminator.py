# Copyright 2022 Huawei Technologies Co., Ltd
# reference by: https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/net/discriminator.py
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
"""Discriminator"""

import mindspore.nn as nn
from mindspore.common.initializer import Normal
from .instance_norm_2d import InstanceNorm2d


class Discriminator(nn.Cell):
    """
    Discriminator network.

    Args:
        channels (int): Base channel number per layer.
        n_dis (int): The number of discriminator layer.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output from the discriminator network.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> discriminator = Discriminator()
    """

    def __init__(self, channels, n_dis):
        super(Discriminator, self).__init__()
        self.has_bias = False

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
            nn.LeakyReLU(alpha=0.2)
        ]

        for _ in range(1, n_dis):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, pad_mode='same', padding=0,
                          weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, pad_mode='same', padding=0,
                          weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
                InstanceNorm2d(channels * 4, affine=False),
                nn.LeakyReLU(alpha=0.2),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
            InstanceNorm2d(channels, affine=False),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
        ]

        self.discriminator = nn.SequentialCell(layers)

    def construct(self, x):
        """ build network """
        out = self.discriminator(x)
        return out
