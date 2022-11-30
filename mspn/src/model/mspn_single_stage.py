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
"""MSPN Single Stage"""
import mindspore.nn as nn

from src.model.blocks import Bottleneck
from src.model.down_sample import ResNetDownsampleModule
from src.model.up_sample import UpsampleModule


class SingleStageModule(nn.Cell):
    """ Single Stage Module for MSPN

    Args:
        output_channel_num (int): Output Tensor Channels.
        output_shape (tuple): Output Tensor Shape.
        use_skip (bool): Whether to Use Cross Stage Feature Aggregation. Default: False.
        generate_skip (bool): Whether to generate skip connection output tensor. Default: False.
        generate_cross_conv (bool): Whether to apply Convolution for the output Tensor to further feed into next stage \
        Module. Default: False.
        channel_num (int): Interim Tensor Channels. Default: 256.
        zero_init_bn (bool): Whether to Use zero initialization On Weight of Batch Normalization. Default: False.

    Inputs:
        Tensor, skip_tensor of previous MSPN Stage
        - **x** (Tensor) - A tensor to be inputted into MSPN Single-stage Module, which should have four dimensions \
        with data type float32.
        - **skip_tensor_1** (Tensor) - A skip tensor from previous MSPN stage, which should have four dimensions with \
        data type float32.
        - **skip_tensor_2** (Tensor) - A skip tensor from previous MSPN stage, which should have four dimensions with \
        data type float32.

    Outputs:
        Tuple of 4 Tensor, the processed tensors.

        - **res** (Tensor) - A tensor which is the output of MSPN Single-stage Module, has the same dimension and data \
        type as `x`.
        - **skip_tensor_1** (Tensor) - A tensor which is the output of the downsample module, has the same dimension \
        and data type as `x`.
        - **skip_tensor_2** (Tensor) - A tensor which is the output of the upsample module, has the same dimension \
        and data type as `x`.
        - **cross_conv** (Tensor) - A tensor which is used for cross stage feature aggregation, has the same dimension \
        and data type as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> single_stage = SingleStageModule(128, (256, 256), use_skip=False, generate_skip=False)
    """
    def __init__(self,
                 output_channel_num: int,
                 output_shape: tuple,
                 use_skip: bool = False,
                 generate_skip: bool = False,
                 generate_cross_conv: bool = False,
                 channel_num: int = 256,
                 zero_init_bn: bool = False
                 ) -> None:
        super(SingleStageModule, self).__init__()
        self.use_skip = use_skip
        self.generate_skip = generate_skip
        self.generate_cross_conv = generate_cross_conv
        self.channel_num = channel_num
        self.zero_init_bn = zero_init_bn
        self.layers = [3, 4, 6, 3]
        self.downsample = ResNetDownsampleModule(Bottleneck, self.layers, self.use_skip, self.zero_init_bn)
        self.upsample = UpsampleModule(output_channel_num, output_shape, self.channel_num, self.generate_skip,
                                       self.generate_cross_conv)

    def construct(self, x, skip_tensor_1, skip_tensor_2):
        """Construct Func"""
        x4, x3, x2, x1 = self.downsample(x, skip_tensor_1, skip_tensor_2)
        res, skip_tensor_1, skip_tensor_2, cross_conv = self.upsample(x4, x3, x2, x1)

        return res, skip_tensor_1, skip_tensor_2, cross_conv
