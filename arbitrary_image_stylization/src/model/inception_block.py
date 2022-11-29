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
"""Several inception blocks of Inception_v3 network."""

from mindspore import nn, ops

class InceptionA(nn.Cell):
    """
    Args:
        num_channels (int): Input channel number
        c1 (int): Out_channels of branch_1.
        c2 (list[int]): Out_channels of every layer in branch_2.
        c3 (list[int]): Out_channels of every layer in branch_3.
        c4 (int): Out_channels of branch_4

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape : math:`(N, C_{out}, H_{out}, W_{out}).

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> block = InceptionA(192, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=32)
    """
    def __init__(self, num_channels: int, c1: int, c2, c3, c4: int):
        super(InceptionA, self).__init__()
        self.conv1 = BasicConv2d(num_channels, c1, kernel_size=1, pad_mode='same')
        self.conv2 = nn.SequentialCell([
            BasicConv2d(num_channels, c2[0], kernel_size=1),
            BasicConv2d(c2[0], c2[1], kernel_size=5)
        ])
        self.conv3 = nn.SequentialCell([
            BasicConv2d(num_channels, c3[0], kernel_size=1),
            BasicConv2d(c3[0], c3[1], kernel_size=3),
            BasicConv2d(c3[1], c3[2], kernel_size=3)
        ])
        self.conv4 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
            BasicConv2d(num_channels, c4, kernel_size=1)
        ])

    def construct(self, x):
        op = ops.Concat(axis=1)
        return op((self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)))

class InceptionB(nn.Cell):
    """
    Args:
        num_channels (int): input channel number
        c1 (int): out_channels of branch_1.
        c2 (list[int]): out_channels of every layer in branch_2.

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape : math:`(N, C_{out}, H_{out}, W_{out}).

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> block = InceptionB(288, c1=384, c2=[64, 96, 96])
    """
    def __init__(self, num_channels: int, c1: int, c2):
        super(InceptionB, self).__init__()
        self.conv1 = BasicConv2d(num_channels, c1, kernel_size=3, stride=2, pad_mode='valid')
        self.conv2 = nn.SequentialCell([
            BasicConv2d(num_channels, c2[0], kernel_size=1),
            BasicConv2d(c2[0], c2[1], kernel_size=3),
            BasicConv2d(c2[1], c2[2], kernel_size=3, stride=2, pad_mode='valid'),
        ])
        self.conv3 = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        ])

    def construct(self, x):
        op = ops.Concat(axis=1)
        return op((self.conv1(x), self.conv2(x), self.conv3(x)))

class InceptionC(nn.Cell):
    """
    Args:
        num_channels (int): input channel number
        c1 (int): out_channels of branch_1.
        c2 (list[int]): out_channels of every layer in branch_2.
        c3 (list[int]): out_channels of every layer in branch_3.
        c4 (int): out_channels of branch_4

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape : math:`(N, C_{out}, H_{out}, W_{out}).

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> block = InceptionC(768, c1=192, c2=[128, 128, 192], c3=[128, 128, 128, 128, 192], c4=192)
    """
    def __init__(self, num_channels: int, c1: int, c2, c3, c4: int):
        super(InceptionC, self).__init__()
        self.conv1 = BasicConv2d(num_channels, c1, kernel_size=1)
        self.conv2 = nn.SequentialCell([
            BasicConv2d(num_channels, c2[0], kernel_size=1),
            BasicConv2d(c2[0], c2[1], kernel_size=(1, 7)),
            BasicConv2d(c2[1], c2[2], kernel_size=(7, 1)),
        ])
        self.conv3 = nn.SequentialCell([
            BasicConv2d(num_channels, c3[0], kernel_size=1),
            BasicConv2d(c3[0], c3[1], kernel_size=(7, 1)),
            BasicConv2d(c3[1], c3[2], kernel_size=(1, 7)),
            BasicConv2d(c3[2], c3[3], kernel_size=(7, 1)),
            BasicConv2d(c3[3], c3[4], kernel_size=(1, 7))
        ])
        self.conv4 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
            BasicConv2d(num_channels, c4, kernel_size=1)
        ])

    def construct(self, x):
        op = ops.Concat(axis=1)
        return op((self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)))

class BasicConv2d(nn.Cell):
    """
    Convalution operation followed by batch_norm layer and activation function.

    Args:
        in_channels (int): Input channels of convolution.
        out_channels (int): Output channels of convolution.

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> conv = BasicConv2d(in_channels=3, out_channels=32, kernel_size=3)
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.activation_fn = nn.ReLU6()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        return x
