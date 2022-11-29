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

"Stem, the first module in Lite-HRNet"

import mindspore as ms
import mindspore.nn as nn

from backbone.funcs import channel_shuffle, channel_split


class Stem(nn.Cell):
    """
    Stem Module

    Args:
        in_channels (int): Input channel size.
        stem_channels (int): Output channel size of conv1 layer.
        out_channels (int): Output channel size.
        expand_ratio (int): Intermediate layer output channel size.

    Inputs:
        - **features** (Tensor) - Input feature tensor.

    Outputs:
        - **output_features** (Tensor) - Output features.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        stem_module = Stem(3, 5, 10 , 2)
        output_features = stem_module(mindspore.Tensor(np.random.rand(4, 3, 256, 192), mindspore.float32))
    """

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.SequentialCell(
            [nn.Conv2d(in_channels,
                       stem_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       pad_mode="pad",
                       has_bias=False),
             nn.BatchNorm2d(stem_channels),
             nn.ReLU()]
        )

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.SequentialCell(
            [nn.Conv2d(branch_channels,
                       branch_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       group=branch_channels,
                       pad_mode="pad",
                       has_bias=False),
             nn.BatchNorm2d(branch_channels),
             nn.Conv2d(branch_channels,
                       inc_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       has_bias=False),
             nn.BatchNorm2d(inc_channels),
             nn.ReLU()]
        )

        self.expand_conv = nn.SequentialCell(
            [nn.Conv2d(branch_channels,
                       mid_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       has_bias=False),
             nn.BatchNorm2d(mid_channels),
             nn.ReLU()]
        )
        self.depthwise_conv = nn.SequentialCell(
            [nn.Conv2d(mid_channels, mid_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       group=mid_channels,
                       pad_mode="pad",
                       has_bias=False),
             nn.BatchNorm2d(mid_channels)]
        )
        self.linear_conv = nn.SequentialCell(
            [nn.Conv2d(mid_channels,
                       branch_channels if stem_channels == self.out_channels else stem_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       has_bias=False),
             nn.BatchNorm2d(branch_channels),
             nn.ReLU()]
        )

    def construct(self, x):
        """Construct"""

        def _inner_construct(x):
            x = self.conv1(x)
            x1, x2 = channel_split(x, 2, axis=1)
            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)
            out = ms.numpy.concatenate((self.branch1(x1), x2), axis=1)
            out = channel_shuffle(out, 2)

            return out


        out = _inner_construct(x)

        return out
