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
"""pose_encoder."""

import numpy as np
from mindspore import nn, ops


class PoseEncoder(nn.Cell):
    """
    A pose encoder to extract style blocks.

    Args:
        image_size (int): The image size. Default: 256.
        channel_base (int): Base channel. Default: 32768.

    Inputs:
        - **x** (Tensor) - A 17x64x64 heatmap.
        - **only_4** (bool) - Only return x4. Default: False.

    Outputs:
        list, 64x64, 32x32, 16x16, 8x8 and 4x4 tensors to input at multiple style blocks, starting from the first.
        Tensor, the 4th output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> model_P = PoseEncoder()
        >>> pose_enc = model_P(mindspore.Tensor(np.random.rand(1, 17, 64, 64), mindspore.float32))
   """
    def __init__(self, image_size=256, channel_base=32768):
        super().__init__()

        self.img_resolution_log2 = int(np.log2(image_size))
        self.img_channels = 3
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels = {res: min(channel_base // res, 512) for res in self.block_resolutions}

        self.conv64x64 = nn.Conv2d(17, channels[64], kernel_size=3, pad_mode='pad', padding=1, has_bias=True)

        self.conv32x32 = nn.CellList([nn.Conv2d(channels[64], channels[32], kernel_size=3, pad_mode="pad",
                                                padding=1, has_bias=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")])

        self.conv16x16 = nn.CellList([nn.Conv2d(channels[32], channels[16], kernel_size=3, pad_mode="pad",
                                                padding=1, has_bias=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")])

        self.conv8x8 = nn.CellList([nn.Conv2d(channels[16], channels[8], kernel_size=3, pad_mode="pad",
                                              padding=1, has_bias=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")])

        self.conv4x4 = nn.CellList([nn.Conv2d(channels[8], channels[4], kernel_size=3, pad_mode="pad",
                                              padding=1, has_bias=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")])

        self.conv4x4_2 = nn.Conv2d(channels[4], channels[4], kernel_size=3, pad_mode="pad", padding=1, has_bias=True)
        self.act = nn.LeakyReLU(alpha=0.01)

    def construct(self, x, only4=False):
        """pose_encoder construct"""
        div = ops.Div()
        ret = []
        x64 = self.act(self.conv64x64(x))
        ret.append(x64)

        x32 = x64
        for layer in self.conv32x32:
            x32 = self.act(layer(x32))
        ret.append(x32)

        x16 = x32
        for layer in self.conv16x16:
            x16 = self.act(layer(x16))
        ret.append(x16)

        x8 = x16
        for layer in self.conv8x8:
            x8 = self.act(layer(x8))
        ret.append(x8)

        x4 = x8
        for layer in self.conv4x4:
            x4 = self.act(layer(x4))
        x4 = x4 + self.act(self.conv4x4_2(x4))
        x4 = div(x4, 1.414)
        ret.append(x4)

        if only4:
            return x4
        return ret
