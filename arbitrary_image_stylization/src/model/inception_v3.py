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
"""
Implementation of Inception_v3 network for encodering style images.
"""

from mindspore import nn

from model.inception_block import InceptionA, InceptionB, InceptionC, BasicConv2d

class InceptionV3(nn.Cell):
    """
    Compared to the original Inception_v3 model,
    some redundant layers that are not used in style prediction model are removed.

    Args:
        in_channels (int): number of input channels. Default: 3.

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape : math:`(N, 768, H_{out}, W_{out}).

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> encoder = InceptionV3()
    """

    def __init__(self, in_channels=3):
        super(InceptionV3, self).__init__()

        self.model = nn.SequentialCell([
            # 299 * 299 * 3
            BasicConv2d(in_channels, out_channels=32, kernel_size=3, stride=2, pad_mode="valid"),
            nn.ReLU(),
            # 149 * 149 * 32
            BasicConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad_mode="valid"),
            nn.ReLU(),
            # 147 * 147 * 32
            BasicConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad_mode="same"),
            nn.ReLU(),
            # 147 * 147 * 64
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
            # 73 * 73 * 64
            BasicConv2d(in_channels=64, out_channels=80, kernel_size=1, pad_mode="valid"),
            nn.ReLU(),
            # 73 * 73 * 80
            BasicConv2d(in_channels=80, out_channels=192, kernel_size=3, pad_mode="valid"),
            nn.ReLU(),
            # 71 * 71 * 192
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
            # 35 * 35 * 192
            InceptionA(192, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=32),
            # 35 * 35 * 256   5b
            InceptionA(256, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=64),
            # 35 * 35 * 288   5c
            InceptionA(288, c1=64, c2=[48, 64], c3=[64, 96, 96], c4=64),
            # 35 * 35 * 288   5d
            InceptionB(288, c1=384, c2=[64, 96, 96]),
            # 17 * 17 * 768   6a
            InceptionC(768, c1=192, c2=[128, 128, 192], c3=[128, 128, 128, 128, 192], c4=192),
            # 17 * 17 * 768   6b
            InceptionC(768, c1=192, c2=[160, 160, 192], c3=[160, 160, 160, 160, 192], c4=192),
            # 17 * 17 * 768   6c
            InceptionC(768, c1=192, c2=[160, 160, 192], c3=[160, 160, 160, 160, 192], c4=192),
            # 17 * 17 * 768   6d
            InceptionC(768, c1=192, c2=[192, 192, 192], c3=[192, 192, 192, 192, 192], c4=192),
        ])

    def construct(self, x):
        x = self.model(x)
        return x
