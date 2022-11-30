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
"""Define the convolution fusion module."""

import mindspore.nn as nn


class ConvNormRelu(nn.Cell):
    """
    Convolution fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        config (class): Option class.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are CONSTANT, REFLECT, SYMMETRIC. Default: CONSTANT.
        use_relu (bool): Use relu or not. Default: True.
        padding (int): Pad size, if it is None, it will calculate by kernel_size. Default: None.

    Outputs:
        Tensor, output tensor of module layer.
    """

    def __init__(self,
                 in_planes,
                 out_planes,
                 config,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvNormRelu, self).__init__()
        norm = nn.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = nn.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if not padding:
            padding = (kernel_size - 1) // 2
        if config.pad_mode == 'REFLECT':
            pad_mode = "REFLECT"
        elif config.pad_mode == "SYMMETRIC":
            pad_mode = "SYMMETRIC"
        if pad_mode == 'CONSTANT':
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad',
                             has_bias=has_bias, padding=padding)
            layers = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layers = [pad, conv, norm]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
