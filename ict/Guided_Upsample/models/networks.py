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
"""Upsample network"""

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P


class Generator(nn.Cell):
    """
    Generator for image generation.

    Args:
        residual_blocks (int): The number of residual_blocks. Default: 8

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, residual_blocks: int = 8):
        super(Generator, self).__init__()

        self.encoder = nn.SequentialCell(
            nn.Pad(((0, 0), (0, 0), (3, 3), (3, 3)), mode='REFLECT'),
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, pad_mode='pad', padding=0, has_bias=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.SequentialCell(*blocks)
        self.decoder = nn.SequentialCell(
            nn.Conv2dTranspose(in_channels=256, out_channels=128, kernel_size=4, stride=2, pad_mode='pad', padding=1,
                               has_bias=True),
            nn.ReLU(),

            nn.Conv2dTranspose(in_channels=128, out_channels=64, kernel_size=4, stride=2, pad_mode='pad', padding=1,
                               has_bias=True),
            nn.ReLU(),

            nn.Pad(((0, 0), (0, 0), (3, 3), (3, 3)), mode='REFLECT'),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, pad_mode='pad', padding=0, has_bias=True),
        )

    def construct(self, images, edges, masks):
        images_masked = (images * P.Cast()((1 - masks), mindspore.float32)) + masks
        x = P.Concat(axis=1)((images_masked, edges))
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (P.Tanh()(x) + 1) / 2

        return x


class ResnetBlock(nn.Cell):
    """
        A resnet block with pad.

        Args:
            dim (int): Input and output channel.
            dilation (int): The size of pad. Default: 1

        Returns:
            Tensor, output tensor.
        """

    def __init__(self, dim: int, dilation: int = 1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.SequentialCell([
            nn.Pad(((0, 0), (0, 0), (dilation, dilation), (dilation, dilation)), mode='REFLECT'),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, pad_mode='pad', dilation=dilation,
                      has_bias=True),
            nn.ReLU(),

            nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode='SYMMETRIC'),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, pad_mode='pad', dilation=1, has_bias=True)
        ])

    def construct(self, x):
        out = x + self.conv_block(x)
        return out


class Discriminator(nn.Cell):
    """
    Discriminator for image generation.

    Args:
        in_channels (int): Input channel.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_channels: int):
        super(Discriminator, self).__init__()

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, pad_mode='pad', padding=1),
            nn.LeakyReLU()
        )

        self.conv2 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad_mode='pad', padding=1),
            nn.LeakyReLU()
        )

        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, pad_mode='pad', padding=1),
            nn.LeakyReLU()
        )

        self.conv4 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, pad_mode='pad', padding=1),
            nn.LeakyReLU()
        )

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, pad_mode='pad', padding=1)

    def construct(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        outputs = P.Sigmoid()(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]
        