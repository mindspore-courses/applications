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
"""Structure of Discriminator"""

import mindspore.nn as nn

from src.util.util import init_weights

__all__ = ["get_discriminator"]

class Discriminator(nn.Cell):
    """
    Structure of Discriminator.

    Args:
        imgae_size (int): The size of training image.

    Inputs:
        - **x** (Tensor) - The image data. The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **out** (Tensor) - The discrimination result. The output has the shape (batchsize, result).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from discriminator import _make_layer, conv2d_cfg
        >>> discriminator = Discriminator(96)
        >>> image = Tensor(np.zeros([1, 3, 96, 96]),mstype.float32)
        >>> out = discriminator(image)
        >>> print(out)
        [[0.5]]
    """
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        feature_map_size = int(image_size // 16)
        self.features = _make_layer(conv2d_cfg)
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(512 * feature_map_size * feature_map_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, 1),
            nn.Sigmoid()
        )

    def construct(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

def get_discriminator(image_size, init_gain):
    """
    Return discriminator by args.

    Args:
        imgae_size (int): The size of training image.
        init_gain (float): The standard deviation to use when initializing network weights.

    Returns:
        nn.Cell, Discriminator with initializd network weights.
    """
    net = Discriminator(image_size)
    init_weights(net, 'normal', init_gain)
    return net

conv2d_cfg = {
    "in": [3, 64, 64, 128, 128, 256, 256, 512],
    "out": [64, 64, 128, 128, 256, 256, 512, 512]
}

def _make_layer(cfg):
    """
    Make stage network of discriminator.

    Args:
        cfg (dict): The number of channels of each layer.

    Returns:
        nn.SequentialCell, Stage network of discriminator.
    """

    layers = []
    stride_add = 0

    for chan_in, chan_out in zip(cfg["in"], cfg["out"]):
        conv2d = nn.Conv2d(chan_in, chan_out, kernel_size=3, stride=1+stride_add, padding=1, pad_mode='pad')
        layers += [conv2d, nn.LeakyReLU(0.2)]
        stride_add = (stride_add + 1) % 2

    return nn.SequentialCell(layers)
