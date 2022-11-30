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
# ==============================================================================
"""Blocks for MSPose"""
from typing import Optional

import mindspore.nn as nn


class ConvBlock(nn.Cell):
    """ Basic Convolutional Block for MSPN

    Args:
        in_channels (int): Input Tensor Channels.
        out_channels (int): Output Tensor Channels.
        kernel_size (int): Convolutional Kernel Size.
        stride (int): Convolutional Stride.
        padding (int): Convolutional Padding.
        use_bn (bool): Whether to Use Batch Normalization. Default: True.
        use_relu (bool): Whether to Use ReLU. Default: True.

    Inputs:
        - **x** (Tensor) - A tensor to be inputted into convolutional block, which should have four dimensions with \
        data type float32.

    Outputs:
        One Tensor, the processed tensor.

        - **x** (Tensor) - A tensor which is processed by convolutional block, has the same dimension and data type \
        as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> conv_block = ConvBlock(3, 16, kernel_size=1, stride=1, padding=0, use_bn=True, use_relu=True)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 use_bn: bool = True,
                 use_relu: bool = True
                 ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode='pad',
                              padding=padding, has_bias=True)
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        """"Construct Func"""
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)

        return x


class Bottleneck(nn.Cell):
    """ Residual Block for MSPN

    Args:
        in_channels (int): Input Tensor Channels.
        channels (int): Output Tensor Channels.
        stride (int): Convolutional Stride. Default: 1.
        downsample (nn.Cell, optional): Downsample Module Implemented Under the Wrap of nn.Cell. Default: None.

    Inputs:
        - **x** (Tensor) - A tensor to be inputted into bottleneck residual block, which should have four dimensions \
        with data type float32.

    Outputs:
        One Tensor, the processed tensor.

        - **conv_block_out** (Tensor) - A tensor which is processed by bottleneck residual block, has the same \
        dimension and data type as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> bottleneck = Bottleneck(3, 16, stride=1, downsample=None)
    """
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Cell] = None
                 ) -> None:
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = ConvBlock(in_channels, channels, kernel_size=1, stride=1, padding=0, use_bn=True,
                                       use_relu=True)
        self.conv_bn_relu2 = ConvBlock(channels, channels, kernel_size=3, stride=stride, padding=1, use_bn=True,
                                       use_relu=True)
        self.conv_bn_relu3 = ConvBlock(channels, channels * self.expansion, kernel_size=1, stride=1, padding=0,
                                       use_bn=True, use_relu=False)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def construct(self, x):
        """Construct Func"""
        conv_block_out = self.conv_bn_relu1(x)
        conv_block_out = self.conv_bn_relu2(conv_block_out)
        conv_block_out = self.conv_bn_relu3(conv_block_out)

        if self.downsample:
            x = self.downsample(x)

        conv_block_out += x
        conv_block_out = self.relu(conv_block_out)

        return conv_block_out


class ResNetTop(nn.Cell):
    """ First Module of MSPN

    Inputs:
        - **x** (Tensor) - A tensor to be inputted into ResNetTop Convolutional block, which should have four \
        dimensions with data type float32.

    Outputs:
        One Tensor, the processed tensor.

        - **out** (Tensor) - A tensor which is processed by ResNetTop Convolutional block, has the same dimension and \
        data type as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> res_top = ResNetTop()
    """
    def __init__(self) -> None:
        super(ResNetTop, self).__init__()
        self.conv = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3, use_bn=True, use_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

    def construct(self, x):
        """Construct Func"""
        out = self.conv(x)
        out = self.maxpool(out)

        return out
