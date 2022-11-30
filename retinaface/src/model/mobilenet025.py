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
"""MobileNet025 with multiple layers of output."""

from mindspore import nn


def conv_dw(in_channel, out_channel, stride, leaky=0.1):
    """
    DepthWise convolution of mobilenet.

    Args:
        in_channel(int): The number of input channel.
        out_channel(int): The number of output channel.
        stride(int): The stride number of the depth-wise convolution.
        leaky(float): The alpha of LeakyReLU. Default: 0.1.

    Returns:
        A nn.SequentialCell, represents the depth-wise convolution layer of MobileNet.
    """
    return nn.SequentialCell([
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=stride,
                  pad_mode='pad', padding=1, group=in_channel, has_bias=False),
        nn.BatchNorm2d(num_features=in_channel, momentum=0.9),
        nn.LeakyReLU(alpha=leaky),
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
                  pad_mode='pad', padding=0, has_bias=False),
        nn.BatchNorm2d(num_features=out_channel, momentum=0.9),
        nn.LeakyReLU(alpha=leaky),
    ])


def conv_bn(in_channel, out_channel, stride=1, leaky=0):
    """
    PointWise convolution of mobilenet.

    Args:
        in_channel(int): The number of input channel.
        out_channel(int): The number of output channel.
        stride(int): The stride number of the point-wise convolution. Default: 1.
        leaky(float): The alpha of LeakyReLU. Default: 0.

    Returns:
        A nn.SequentialCell, represents the point-wise convolution layer of MobileNet.
    """
    return nn.SequentialCell([
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                  pad_mode='pad', padding=1, has_bias=False),
        nn.BatchNorm2d(num_features=out_channel, momentum=0.9),
        nn.LeakyReLU(alpha=leaky)
    ])


class MobileNetV1(nn.Cell):
    """
    The MobileNetV1 network structure.

    Inputs:
        - **x** (Tensor) - An input tensor, which shape is :math:`(B, in_channel, H, W)`.

    Outputs:
        Tuple of 3 Tensor, which contains output Tensor from top, middle and bottom layer.

        - **tensor1** (Tensor) - Input tensor of the top layer, which shape is :math:`(B, 64, H/8, W/8)`.
        - **tensor2** (Tensor) - Input tensor of the medium layer, which shape is :math:`(B, 128, H/16, W/16)`.
        - **tensor3** (Tensor) - Input tensor of the bottom layer, which shape is :math:`(B, 256, H/32, W/32)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from src.model.mobilenet025 import MobileNetV1
        >>> from mindspore import ops
        >>> mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
        >>> mobilenet = MobileNetV1()
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 3, 128, 128), mindspore.float32)
        >>> result = mobilenet(x)
        >>> print(result[0].shape, result[1].shape, result[2].shape)
        (1, 64, 16, 16) (1, 128, 8, 8) (1, 256, 4, 4)
    """

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.SequentialCell([
            conv_bn(3, 8, 2, leaky=0.1),
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        ])
        self.stage2 = nn.SequentialCell([
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        ])
        self.stage3 = nn.SequentialCell([
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        ])

    def construct(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x1, x2, x3
