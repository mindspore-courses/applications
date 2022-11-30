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
"""Define faceAlignment model"""
from typing import List

import mindspore.nn as nn

network_config = [
    # in_channels, out_channels, kernel_size, stride, padding, dilation, group
    [3, 16, 3, 2, 1, 1, 1],
    [16, 16, 3, 1, 1, 1, 16],
    [16, 32, 1, 1, 0, 1, 1],
    [32, 32, 3, 2, 1, 1, 32],
    [32, 64, 1, 1, 0, 1, 1],
    [64, 64, 3, 1, 1, 1, 64],
    [64, 64, 1, 1, 0, 1, 1],
    [64, 64, 3, 2, 1, 1, 64],
    [64, 128, 1, 1, 0, 1, 1],
    [128, 128, 3, 1, 1, 1, 128],
    [128, 128, 1, 1, 0, 1, 1],
    [128, 128, 3, 2, 1, 1, 128],
    [128, 256, 1, 1, 0, 1, 1],
    [256, 256, 3, 1, 1, 1, 256],
    [256, 256, 1, 1, 0, 1, 1],
    [256, 256, 3, 1, 1, 1, 256],
    [256, 256, 1, 1, 0, 1, 1],
    [256, 256, 3, 1, 1, 1, 256],
    [256, 256, 1, 1, 0, 1, 1],
    [256, 256, 3, 1, 1, 1, 256],
    [256, 256, 1, 1, 0, 1, 1],
    [256, 256, 3, 1, 1, 1, 256],
    [256, 256, 1, 1, 0, 1, 1],
    [256, 256, 3, 2, 1, 1, 256],
    [256, 512, 1, 1, 0, 1, 1],
    [512, 512, 3, 1, 1, 1, 512],
    [512, 512, 1, 1, 0, 1, 1],
    [512, 64, 3, 2, 1, 1, 1]
]


class Facealignment2d(nn.Cell):
    """
    Model define for 2D face alignment work
    Model structure and layer names are directly translated from the given ONNX file

    Args:
        output_channel (int) - Should be number of alignment points * 2, this input is 388 for Helen dataset.

    Inputs:
        X(Tensor(1, 3, 192, 192)): Input image in tensor

    Outputs:
        x(Tensor(1, 1, output_channel)): Predict output. Each point takes 2 channels.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, output_channel):
        super(Facealignment2d, self).__init__()
        self.network_config = network_config
        self.features = self._make_layer(network_config, output_channel)

    def construct(self, x):
        """
        Define forward pass
        """
        x = self.features(x)
        return x

    def _make_layer(self, cfg: List[List[int]], output_channel: int) -> nn.SequentialCell:
        '''
        Make layer for model 'FaceAlignment2d'.

        Args:
            cfg: Model layer config, like 'network_config' above
            output_channel(int) : Should be number of alignment points * 2, this input is 388 for Helen dataset.

        Returns:
            SequentialCell, Contains layers generated With 'cfg'

        Examples:
            >>>_make_layer(network_config, 388)
        '''
        layers = []
        for v in cfg:
            layers += [nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                 kernel_size=v[2], stride=v[3],
                                 pad_mode="pad",
                                 padding=(v[4], v[4], v[4], v[4]),
                                 dilation=v[5], group=v[6],
                                 has_bias=False),
                       nn.BatchNorm2d(num_features=v[1], eps=1e-3),
                       nn.PReLU(channel=v[1], w=0.25)]
        out_channels = cfg[-1][1] * cfg[-1][2] * cfg[-1][2]
        layers += [nn.Flatten(), nn.Flatten(), nn.Dense(in_channels=out_channels, out_channels=output_channel)]
        return nn.SequentialCell(layers)
