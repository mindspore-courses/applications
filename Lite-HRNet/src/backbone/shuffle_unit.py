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

"""Shuffle unit in Naive Lite-HRNet"""

import mindspore as ms
import mindspore.nn as nn

from backbone.funcs import channel_shuffle, channel_split


class ShuffleUnit(nn.Cell):
    """Inverted Residual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The inputs channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride. Default: 1.

    Inputs:
        - **features** (Tensor) - Input feature tensor.

    Outputs:
        - **output_features** (Tensor) - Output features.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> shuffle_unit = ShuffleUnit(3, 3)
        >>> output_features = shuffle_unit(mindspore.Tensor(np.random.rand(16, 3, 64, 48), mindspore.float32))
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super().__init__()
        self.stride = stride

        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.SequentialCell(
                [nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=3,
                           stride=self.stride,
                           padding=1,
                           group=in_channels,
                           pad_mode="pad",
                           has_bias=False),
                 nn.BatchNorm2d(in_channels),
                 nn.Conv2d(in_channels,
                           branch_features,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           has_bias=False),
                 nn.BatchNorm2d(branch_features),
                 nn.ReLU()]
            )

        self.branch2 = nn.SequentialCell(
            [nn.Conv2d(in_channels if (self.stride > 1) else branch_features,
                       branch_features,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       has_bias=False),
             nn.BatchNorm2d(branch_features),
             nn.ReLU(),
             nn.Conv2d(branch_features,
                       branch_features,
                       kernel_size=3,
                       stride=self.stride,
                       padding=1,
                       group=branch_features,
                       pad_mode="pad",
                       has_bias=False),
             nn.BatchNorm2d(branch_features),
             nn.Conv2d(branch_features,
                       branch_features,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       has_bias=False),
             nn.BatchNorm2d(branch_features),
             nn.ReLU()]
            )

    def construct(self, x):
        """Construct."""

        def _inner_construct(x):
            if self.stride > 1:
                out = ms.numpy.concatenate((self.branch1(x), self.branch2(x)), axis=1)
            else:
                x1, x2 = channel_split(x, groups=2, axis=1)
                out = ms.numpy.concatenate((x1, self.branch2(x2)), axis=1)

            out = channel_shuffle(out, 2)

            return out

        out = _inner_construct(x)

        return out
