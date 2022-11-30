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
"""Define some block for CycleGAN"""

import mindspore.nn as nn

from .convrelu import ConvNormReLU


class ResidualBlock(nn.Cell):
    """
    A resnet block is a conv block with skip connections.
    We construct a ResidualBlock with ConvNormReLU, one uses relu, another don't use.

    Args:
        dim (int): Input and output channel.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance". Default: "batch"
        dropout (bool): Use dropout or not. Default: False.
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Returns:
        Tensor, output tensor.
    """
    def __init__(self, dim, norm_mode='batch', dropout=False, pad_mode="CONSTANT"):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode)
        self.conv2 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode, use_relu=False)
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        out = self.conv1(x)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        return x + out
