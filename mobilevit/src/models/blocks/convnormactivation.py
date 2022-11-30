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
"""Blocks for classification."""

from typing import Optional

from mindspore import nn


class ConvNormActivation(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.BatchNorm2d.
        activation (nn.Cell, optional): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> conv = ConvNormActivation(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU
                 ) -> None:
        super(ConvNormActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                pad_mode='pad',
                padding=padding,
                group=groups
            )
        ]

        if norm:
            layers.append(norm(out_planes))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
