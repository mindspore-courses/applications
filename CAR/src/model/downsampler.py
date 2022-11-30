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
"""ResamplerNet. To generate kernel weights and kernel offsets."""

import mindspore.nn as nn
import mindspore.ops as ops

from model.block import MeanShift, NormalizeBySum, DownsampleBlock, ReflectionPad2d, UpsampleBlock, ResidualBlock


class DSN(nn.Cell):
    """
    ResamplerNet, generate kernel weights and kernel offsets.

    Args:
        k_size(int): The kernel size for downscaling.
        input_channels(int): Input feature dimension. Default:3
        scale(int): Downscaling rate. Default:4

    Inputs:
        x(Tensor): Input tensor. The shape is (b, input_channels, h, w).

    Returns:
        kernels(Tensor): The shape is (b, k_size^2, h // scale , w // scale).
        offsets_h(Tensor): The shape is (b, k_size^2, h // scale , w // scale).
        offsets_v(Tensor): The shape is (b, k_size^2, h // scale , w // scale).
    """

    def __init__(self, k_size, input_channels=3, scale=4):
        super().__init__()

        self.k_size = k_size
        self.sub_mean = MeanShift(1)
        self.normalize = NormalizeBySum()

        self.ds_1 = nn.SequentialCell(
            ReflectionPad2d(2),
            nn.Conv2d(input_channels, 64, 5, pad_mode="valid", has_bias=True),
            nn.LeakyReLU(0.2),
        )

        self.ds_2 = DownsampleBlock(2, 64, 128, ksize=1)
        self.ds_4 = DownsampleBlock(2, 128, 128, ksize=1)

        res_4 = []
        for _ in range(5):
            res_4 += [ResidualBlock(128, 128)]
        self.res_4 = nn.SequentialCell(*res_4)

        self.ds_8 = DownsampleBlock(2, 128, 256)

        self.kernels_trunk = nn.SequentialCell(
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            UpsampleBlock(8 // scale, 256, 256, ksize=1),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="pad", has_bias=True),
            nn.ReLU()
        )

        self.kernels_weight = nn.SequentialCell(
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="pad", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, k_size**2, 3, pad_mode="pad", has_bias=True),
        )

        self.offsets_trunk = nn.SequentialCell(
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            UpsampleBlock(8 // scale, 256, 256, ksize=1),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="valid", has_bias=True),
            nn.ReLU(),
        )

        self.offsets_h_generation = nn.SequentialCell(
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, k_size**2, 3, pad_mode="valid", has_bias=True),
            nn.Tanh(),
        )

        self.offsets_v_generation = nn.SequentialCell(
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            ReflectionPad2d(1),
            nn.Conv2d(256, k_size**2, 3, pad_mode="valid", has_bias=True),
            nn.Tanh(),
        )

    def construct(self, img):
        """ build network """

        x = self.sub_mean(img)

        x = self.ds_1(x)
        x = self.ds_2(x)
        x = self.ds_4(x)
        x = x + self.res_4(x)
        x = self.ds_8(x)

        kt = self.kernels_trunk(x)

        kt = self.kernels_weight(kt)
        k_weight = ops.clip_by_value(kt, 1e-6, 1)
        kernels = self.normalize(k_weight)

        ot = self.offsets_trunk(x)
        offsets_h = self.offsets_h_generation(ot)
        offsets_v = self.offsets_v_generation(ot)

        return kernels, offsets_h, offsets_v
