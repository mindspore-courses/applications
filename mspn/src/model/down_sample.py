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
# ==============================================================================
"""downsample"""
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeNormal

from src.model.blocks import ConvBlock, Bottleneck


class ResNetDownsampleModule(nn.Cell):
    """ Residual Downsample Module for MSPN

    Args:
        block (nn.Cell): Residual Downsample Module Implemented Under the Wrap of nn.Cell.
        layer_num_list (list): Num of Stacking Residual Blocks.
        use_skip (bool): Whether to Use Cross Stage Feature Aggregation. Default: False.
        zero_init_bn (bool): Whether to Use zero initialization On Weight of Batch Normalization. Default: False.

    Inputs:
        - **x** (Tensor) - A tensor to be inputted into ResNetDownsampleModule block, which should have four \
        dimensions with data type float32.
        - **skip_tensor_1** (Tensor) - A skip tensor from previous MSPN stage, which should have four dimensions with \
        data type float32.
        - **skip_tensor_2** (Tensor) - A skip tensor from previous MSPN stage, which should have four dimensions with \
        data type float32.

    Outputs:
        Tuple of 4 Tensor, the processed tensors.

        - **x4** (Tensor) - A tensor which is the output of the first downsample sub-module, has the same dimension \
        and data type as `x`.
        - **x3** (Tensor) - A tensor which is the output of the second downsample sub-module, has the same dimension \
        and data type as `x`.
        - **x2** (Tensor) - A tensor which is the output of the third downsample sub-module, has the same dimension \
        and data type as `x`.
        - **x1** (Tensor) - A tensor which is the output of the fourth downsample sub-module, has the same dimension \
        and data type as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> downsample = ResNetDownsampleModule(Bottleneck, layer_num_list=[3, 4, 6, 3])
    """
    def __init__(self,
                 block: nn.Cell,
                 layer_num_list: list,
                 use_skip: bool = False,
                 zero_init_bn: bool = False
                 ) -> None:
        super(ResNetDownsampleModule, self).__init__()
        self.use_skip = use_skip
        self.in_channels = 64
        self.layer1 = self._make_layer(block, 64, layer_num_list[0])
        self.layer2 = self._make_layer(block, 128, layer_num_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_num_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_num_list[3], stride=2)

        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), m.weight.shape, mindspore.float32)
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = initializer(1, m.gamma.shape, mindspore.float32)
                m.beta = initializer(0, m.beta.shape, mindspore.float32)

        if zero_init_bn:
            for _, m in self.cells_and_names():
                if isinstance(m, Bottleneck):
                    m.bn3.weight = initializer(0, m.bn3.weight.shape, mindspore.float32)

    def _make_layer(self, block, channels, num_blocks, stride=1):
        """Stacking Residual Blocks"""
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = ConvBlock(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride,
                                   padding=0, use_bn=True, use_relu=False)

        layers = list()
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.SequentialCell(*layers)

    def construct(self, x, skip_tensor_1, skip_tensor_2):
        """Construct Func"""
        x1 = self.layer1(x)
        if self.use_skip:
            x1 = x1 + skip_tensor_1[0] + skip_tensor_2[0]
        x2 = self.layer2(x1)
        if self.use_skip:
            x2 = x2 + skip_tensor_1[1] + skip_tensor_2[1]
        x3 = self.layer3(x2)
        if self.use_skip:
            x3 = x3 + skip_tensor_1[2] + skip_tensor_2[2]
        x4 = self.layer4(x3)
        if self.use_skip:
            x4 = x4 + skip_tensor_1[3] + skip_tensor_2[3]

        return x4, x3, x2, x1
