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
"""Build vgg19 network."""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import initializer


def _make_layer(base, batch_norm):
    """
    Build the VGG network based on the parameters in the list.

    Args:
        base (list): Vgg network layer definition.The numbers in the list represent the number of features.'m'
            stands for maxpool2d.
        batch_norm (bool): Whether to use batch normalization.

    Returns:
        Cell, complete vgg network.
    """

    layers = []
    in_channels = 3
    i = 0
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight_shape = (v, in_channels, 3, 3)
            weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=0,
                               pad_mode='same',
                               has_bias=False,
                               weight_init=weight)
            if batch_norm:
                if i == len(base) - 1:
                    layers += [conv2d, nn.BatchNorm2d(v)]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                if i == len(base) - 1:
                    layers += [conv2d]
                else:
                    layers += [conv2d, nn.ReLU()]
            in_channels = v
        i += 1
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        include_top (bool): Whether to include the 3 fully-connected layers at the top of the network. Default: True.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output from the VGG network.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> vgg = Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        ...       num_classes=1000, batch_norm=False)
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, include_top=True):
        super(Vgg, self).__init__()
        self.layers = _make_layer(base, batch_norm=batch_norm)
        self.include_top = include_top
        self.flatten = nn.Flatten()
        dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])

    def construct(self, x):
        """ build network """
        x = self.layers(x)
        return x
