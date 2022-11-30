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
"""Inverted residual block."""

import mindspore.nn as nn
from mindspore.common.initializer import Normal

from .conv2d_block import ConvBlock
from .instance_norm_2d import InstanceNorm2d


class InvertedResBlock(nn.Cell):
    """
    Define inverted residual block.

    Args:
        channels (int): Number of input channels. Default: 256.
        out_channels (int): Number of input channels. Default: 256.
        stride (int): Stride size of convolution kernel. Default: 1.
        expand_ratio (int): Number of channels expansion ratio. Default: 2.
        has_bias (bool): Whether to add bias. Default: False.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output from inverted residual block.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> inverted_res_block = InvertedResBlock(128, 256)
    """

    def __init__(self, channels=256, out_channels=256, stride=1, expand_ratio=2, has_bias=False):
        super(InvertedResBlock, self).__init__()
        self.out_channels = out_channels
        self.channels = channels
        self.stride = stride
        bottleneck_dim = round(expand_ratio * channels)
        self.conv_block = ConvBlock(channels, bottleneck_dim, kernel_size=1, stride=1, padding=0, has_bias=has_bias)
        self.depthwise_conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, group=bottleneck_dim, stride=1,
                                        pad_mode='same', padding=0, weight_init=Normal(mean=0, sigma=0.02),
                                        has_bias=has_bias)
        self.conv = nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, stride=1,
                              weight_init=Normal(mean=0, sigma=0.02), has_bias=has_bias)

        self.ins_norm1 = InstanceNorm2d(bottleneck_dim, affine=False)
        self.ins_norm2 = InstanceNorm2d(out_channels, affine=False)
        self.activation = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):
        """ build network """
        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)
        if (self.channels == self.out_channels) and self.stride == 1:
            out = out + x
        return out
