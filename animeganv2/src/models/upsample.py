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
"""Define up-sampling operation."""

import mindspore.nn as nn


class UpSample(nn.Cell):
    """
    Define up-sampling and convolution module.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of input channels.
        kernel_size (int): Convolution kernel size. Default: 3.
        has_bias (bool): Whether to add bias. Default: False.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output upsample.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> upsample = UpSample(128,128)
    """

    def __init__(self, channels, out_channels, kernel_size=3, has_bias=False):
        super(UpSample, self).__init__()

        self.conv = nn.Conv2d(channels, out_channels, stride=1, kernel_size=kernel_size, has_bias=has_bias)
        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        """ build network """
        out = self.resize(x, scale_factor=2)
        out = self.conv(out)

        return out
