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
"""Structure of Generator"""

import math

import mindspore.nn as nn
import mindspore.ops as ops

from src.util.util import init_weights

__all__ = ["get_generator"]

class ResidualBlock(nn.Cell):
    """
    Structure of ResidualBlock.

    Args:
        channels: (int): The number of channels.

    Inputs:
        - **x** (Tensor) - The output of the PReLU(close to the first conv layer) or the last residual blocks. The input
          shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **out** (Tensor) - The ResidualBlock output. The output has the shape (batchsize, num_channels, height,
          width).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> residual = ResidualBlock(3)
        >>> image = Tensor(np.zeros([1, 3, 24, 24]),mstype.float32)
        >>> out = residual(image)
        >>> print(out.shape)
        (1, 3, 24, 24)
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(channels)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return out

class SubpixelConvolutionLayer(nn.Cell):
    """
    Structure of SubpixelConvolutionLayer.

    Args:
        channels: (int): The number of channels.

    Inputs:
        - **x** (Tensor) - The output of the element-wise-sum or sub-pixel convolution layer.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **out** (Tensor) - The SubpixelConvolutionLayer output.
          The output has the shape (batchsize, num_channels, height, width).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> import mindspore.common.dtype as mstype
        >>> subpixel = SubpixelConvolutionLayer(3)
        >>> image = Tensor(np.zeros([1, 3, 24, 24]),mstype.float32)
        >>> out = subpixel(image)
        >>> print(out.shape)
        (1, 3, 48, 48)
    """
    def __init__(self, channels):
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad')
        self.pixel_shuffle = ops.DepthToSpace(2)
        self.prelu = nn.PReLU(channels)

    def construct(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class Generator(nn.Cell):
    """
    Structure of Generator.

    Args:
        upscale_factor (int): Upsampling factor of SRGAN.

    Inputs:
        - **x** (Tensor) - The image data.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **out** (Tensor) - The generation result.
          The output has the shape (batchsize, num_channels, height, width).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import math
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from generator import ResidualBlock, SubpixelConvolutionLayer
        >>> generator = Generator(4)
        >>> image = Tensor(np.zeros([1, 3, 24, 24]),mstype.float32)
        >>> out = generator(image)
        >>> print(out.shape)
        (1, 3, 96, 96)
    """
    def __init__(self, upscale_factor):

        super(Generator, self).__init__()
        # Calculating the number of subpixel convolution layers.
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 2))
        # First layer.
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, has_bias=True, pad_mode='pad'),
            nn.PReLU(channel=64))

        # 16 Residual blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(64))
        self.trunk = nn.SequentialCell(*trunk)

        # Second conv layer post residual blocks.
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad'),
            nn.PReLU(channel=64)
        )

        # 2 Sub-pixel convolution layers.
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(64))
        self.subpixel_conv = nn.SequentialCell(*subpixel_conv_layers)

        # Final output layer.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, has_bias=True, pad_mode='pad')
        self.tanh = nn.Tanh()
    def construct(self, x):
        conv1 = self.conv1(x)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = conv1+conv2
        out = self.subpixel_conv(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out


def get_generator(upscale_factor, init_gain):
    """
    Return generator by args.

    Args:
        upscale_factor (int): Upsampling factor for SRGAN.
        init_gain (float): The standard deviation to use when initializing network weights.

    Returns:
        nn.Cell, Generator with initializd network weights.
    """
    net = Generator(upscale_factor)
    init_weights(net, 'normal', init_gain)
    return net
