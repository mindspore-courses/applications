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
"""Loss cell."""

import mindspore.nn as nn

from .utils import rgb_to_yuv


class ColorLoss(nn.Cell):
    r"""
    Calculate color loss.

    .. math::

        L_{col}(G,D) = E_{pi\sim S_{data}(p)}\lbrack\left| \left| Y\left( G\left( p_{i} \right) \right) - Y\left( p_{i}
         \right) \right| \right|_{1} + \left| \left| U\left( G\left( p_{i} \right) \right) - U\left( p_{i} \right)
         \right| \right|_{H} + \left| \left| V\left( G\left( p_{i} \right) \right) - V\left( p_{i} \right) \right|
         \right|_{H}\rbrack

    Where，:math:`Y(p_{i})`，:math:`U(p_{i})` and :math:`V(p_{i})` respectively represent the three YUV channels of the
    image :math:`p_{i}`, :math:`H` represents Huber loss.

    Inputs:
        - **image** (tensor) - Real world image.
        - **image_g** (tensor) - Fake anime image.

    Outputs:
        - **color_loss** (tensor) - Color loss.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> color_loss = ColorLoss()
    """

    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def construct(self, image, image_g):
        """ build network """
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)
        color_loss = (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                      self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                      self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))
        return color_loss
