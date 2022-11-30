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
""" MobileNetV2 backbone."""

from typing import Optional, List

from mindspore import nn
from mindspore.ops.operations import Add

from models.blocks.convnormactivation import ConvNormActivation
from models.swish import Swish

__all__ = [
    "InvertedResidual"
]

class InvertedResidual(nn.Cell):
    """
    Mobilenetv2 residual block definition.

    Args:
        in_channel (int): The input channel.
        out_channel (int): The output channel.
        stride (int): The Stride size for the first convolutional layer. Default: 1.
        expand_ratio (int): The expand ration of input channel.
        norm (nn.Cell, optional): The norm layer that will be stacked on top of the convoution
            layer. Default: None.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import InvertedResidual
        >>> InvertedResidual(3, 256, 1, 1)
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int,
                 expand_ratio: int,
                 norm: Optional[nn.Cell] = None,
                 activation: Optional[nn.Cell] = None
                 ) -> None:
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        if not norm:
            norm = nn.BatchNorm2d
        if not activation:
            activation = Swish

        hidden_dim = round(in_channel * expand_ratio)
        self.use_res_connect = stride == 1 and in_channel == out_channel

        layers: List[nn.Cell] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvNormActivation(in_channel, hidden_dim, kernel_size=1, norm=norm, activation=activation))
        layers.extend([
            # dw
            ConvNormActivation(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm=norm,
                activation=activation
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channel, kernel_size=1,
                      stride=1, has_bias=False),
            norm(out_channel)
        ])
        self.conv = nn.SequentialCell(layers)
        self.add = Add()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)
        return x
