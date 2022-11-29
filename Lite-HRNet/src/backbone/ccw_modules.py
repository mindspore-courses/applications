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

"""Cross channel weighting"""

import mindspore as ms
import mindspore.nn as nn

from backbone.funcs import adaptive_avg_pool_2d_func_gpu, channel_shuffle, channel_split


class SpatialWeighting(nn.Cell):
    """
    Spatial Weighting Module

    Args:
        channels (int): Input channels size.
        ratio (int): Define input channel // output channel in conv1. Default: 16

    Inputs:
        -**features** (Tensor) - 4-dim features (batch_size, channel_num, height, width)

    Outputs:
        -**weighted_features** (Tensor) - Weighted features, have the same shape as the input

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sw_module = SpatialWeighting(channels=16, ratio=16)
        >>> sw_output = sw_module(mindspore.Tensor(np.random.rand(4, 16, 64, 64), mindspore.float32))

    """

    def __init__(self,
                 channels,
                 ratio=16):
        super().__init__()
        self.global_avgpool = ms.ops.AdaptiveAvgPool2D((1, 1))
        self.conv1 = nn.SequentialCell(
            [nn.Conv2d(channels,
                       int(channels / ratio),
                       kernel_size=1,
                       stride=1,
                       pad_mode="pad",
                       has_bias=True),
             nn.ReLU()]
        )
        self.conv2 = nn.SequentialCell(
            [nn.Conv2d(int(channels / ratio),
                       channels,
                       kernel_size=1,
                       stride=1,
                       pad_mode="pad",
                       has_bias=True),
             nn.Sigmoid()]
        )

    def construct(self, x):
        """Construct"""

        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Cell):
    """
    Cross resolution weighting module

    Args:
        channels (int): Input channels size.
        ratio (int): Define input channel // output channel in conv1. Default: 16

    Inputs:
        -**features** (Tensor) - 4-dim features (batch_size, channel_num, height, width)

    Outputs:
        -**weighted_features** (Tensor) - Weighted features, have the same shape as the input

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> cr_module = CrossResolutionWeighting(channels=16, ratio=16)
        >>> ccw_output = cr_module(mindspore.Tensor(np.random.rand(4, 16, 64, 64), mindspore.float32))
    """

    def __init__(self,
                 channels,
                 ratio=16):
        super().__init__()
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(total_channel,
                      int(total_channel / ratio),
                      kernel_size=1,
                      stride=1,
                      pad_mode="pad",
                      has_bias=False),
            nn.BatchNorm2d(int(total_channel / ratio)),
            nn.ReLU()
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(int(total_channel / ratio),
                      total_channel,
                      kernel_size=1,
                      stride=1,
                      pad_mode="pad",
                      has_bias=False),
            nn.BatchNorm2d(total_channel),
            nn.Sigmoid()
        )

    def construct(self, x):
        """Construct"""

        mini_size = x[-1].shape[-2:]
        out = []
        for s in x[:-1]:
            out.append(adaptive_avg_pool_2d_func_gpu(inputs=s, output_res=mini_size))

        out.append(x[-1])
        out = ms.numpy.concatenate(out, axis=1)
        out = self.conv1(out)
        out = self.conv2(out)

        out = channel_split(out, self.channels, axis=1)
        output = []
        for s, a in zip(x, out):
            pooled_a = adaptive_avg_pool_2d_func_gpu(inputs=a, output_res=s.shape[-2:])
            output.append(s * pooled_a)

        return output

class ConditionalChannelWeighting(nn.Cell):
    """
    Conditional Channel Weighting Module

    Args:
        in_channels (int): Input channel size.
        stride (int): Stride.
        reduce_ratio: Define input channel // output channel in cross resolution weighting.

    Inputs:
        -**features** (Tensor) - 4-dim features (batch_size, channel_num, height, width).

    Outputs:
        -**weighted_features** (Tensor) - Weighted features, have the same shape as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> ccw_module = ConditionalChannelWeighting(channels=16, stride=1, reduce_ratio=16)
        >>> ccw_output = ccw_module(mindspore.Tensor(np.random.rand(4, 16, 64, 64), mindspore.float32))
    """

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio):
        super().__init__()
        self.stride = stride

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio)

        self.depthwise_convs = nn.CellList([
            nn.SequentialCell(
                [nn.Conv2d(channel, channel,
                           kernel_size=3,
                           stride=self.stride,
                           padding=1,
                           group=channel,
                           pad_mode="pad",
                           has_bias=False),
                 nn.BatchNorm2d(channel)]
            ) for channel in branch_channels
        ])

        self.spatial_weighting = nn.CellList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def construct(self, x):
        """Construct function."""

        x_split = []
        output = []
        for s in x:
            x_split.append(channel_split(s, 2, axis=1))
        x1 = []
        x2 = []
        for s in x_split:
            x1.append(s[0])
            x2.append(s[1])

        x2 = self.cross_resolution_weighting(x2)
        x3 = []
        i = 0
        for s, dw, sw in zip(x2, self.depthwise_convs, self.spatial_weighting):
            ss = dw(s)
            ss = sw(ss)
            x3.append(ss)
            i = i+1
        out = []
        for s1, s3 in zip(x1, x3):
            s13 = ms.numpy.concatenate([s1, s3], axis=1)
            out.append(s13)

        for s in out:
            s_shuffle = channel_shuffle(s, 2)
            output.append(s_shuffle)

        return output
