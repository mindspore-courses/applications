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
"""Build generator and discriminator."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from .network_module import Conv2dLayer, GatedConv2d, TransposeGatedConv2d
from .compute_attention import ContextualAttention, ApplyAttention


class Coarse(nn.Cell):
    """
    Build the first stage of generator: coarse network.

    Return:
        first_out: The output of coarse network.
    """

    def __init__(self):
        super(Coarse, self).__init__()
        self.coarse1 = nn.SequentialCell(
            GatedConv2d(4, 32, 5, 2, 1, sc=True),
            GatedConv2d(32, 32, 3, 1, 1, sc=True),
            GatedConv2d(32, 64, 3, 2, 1, sc=True)
        )
        self.coarse2 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True)
        )
        self.coarse3 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True)
        )
        self.coarse4 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 2, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, sc=True)
        )
        self.coarse5 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 4, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, sc=True)
        )
        self.coarse6 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 8, sc=True),
            GatedConv2d(64, 64, 3, 1, 8, sc=True),
        )
        self.coarse7 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, sc=True),
        )
        self.coarse8 = nn.SequentialCell(
            TransposeGatedConv2d(64, 32, 3, 1, 1, sc=True),
            GatedConv2d(32, 32, 3, 1, 1, sc=True),
            TransposeGatedConv2d(32, 3, 3, 1, 1, sc=True),
        )

    def construct(self, first_in):
        """coarse forward network"""

        first_out = self.coarse1(first_in)
        first_out = self.coarse2(first_out)
        first_out = self.coarse3(first_out)
        first_out = self.coarse4(first_out)
        first_out = self.coarse5(first_out)
        first_out = self.coarse6(first_out)
        first_out = self.coarse7(first_out)
        first_out = self.coarse8(first_out)
        first_out = ops.clip_by_value(first_out, -1, 1)
        return first_out


class GatedGenerator(nn.Cell):
    """
    Build the second stage of generator: refine network and complete generator.

    Args:
        opt(class): option class.

    Return:
        first_out: The output of coarse network.
        second_out: The output of refine network.
        match: Attention score.
    """

    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        self.coarse = Coarse()
        self.refinement1 = nn.SequentialCell(
            GatedConv2d(4, 32, 3, 2, 1),
            GatedConv2d(32, 32, 3, 1, 1)
        )
        self.refinement2 = nn.SequentialCell(
            GatedConv2d(32, 64, 3, 2, 1),
            GatedConv2d(64, 64, 3, 1, 1)
        )
        self.refinement3 = nn.SequentialCell(
            GatedConv2d(64, 128, 3, 2, 1),
            GatedConv2d(128, 128, 3, 1, 1)
        )
        self.refinement4 = GatedConv2d(128, 128, 3, 1, 1)
        self.refinement5 = nn.SequentialCell(
            GatedConv2d(128, 128, 3, 1, 2),
            GatedConv2d(128, 128, 3, 1, 4)
        )
        self.refinement6 = nn.SequentialCell(
            GatedConv2d(128, 128, 3, 1, 8),
            GatedConv2d(128, 128, 3, 1, 16)
        )
        self.refinement7 = nn.SequentialCell(
            TransposeGatedConv2d(128, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1)
        )
        self.refinement8 = nn.SequentialCell(
            TransposeGatedConv2d(128, 32, 3, 1, 1),
            GatedConv2d(32, 32, 3, 1, 1)
        )
        self.refinement9 = TransposeGatedConv2d(64, 3, 3, 1, 1)
        self.conv_att1 = GatedConv2d(128, 128, 3, 1, 1)
        self.conv_att2 = GatedConv2d(256, 128, 3, 1, 1)
        self.batch = opt.train_batchsize
        self.apply_attention1 = ApplyAttention([self.batch, 64, 128, 128], [self.batch, 1024, 32, 32])
        self.apply_attention2 = ApplyAttention([self.batch, 32, 256, 256], [self.batch, 1024, 32, 32])
        self.ones = ops.Ones()
        self.concat = ops.Concat(1)
        self.bilinear_256 = ops.ResizeBilinear((256, 256))
        self.bilinear_512 = ops.ResizeBilinear((512, 512))
        self.reshape = ops.Reshape()
        self.contextual_attention = ContextualAttention(fuse=True, dtype=mindspore.float32)
        self.cat = ops.Concat(1)
        self.method = opt.attention_type

    def construct(self, img, mask):
        """generator forward network"""

        x_in = img.astype(mindspore.float32)
        shape = x_in.shape
        mask_batch = self.ones((shape[0], 1, shape[2], shape[3]), mindspore.float32)
        mask_batch = mask_batch * mask
        first_in = self.concat((x_in, mask_batch))
        first_in = self.bilinear_256(first_in)
        first_out = self.coarse(first_in)
        first_out = self.bilinear_512(first_out)
        first_out = self.reshape(first_out, (shape[0], shape[1], shape[2], shape[3]))
        x_coarse = first_out * mask_batch + x_in * (1. - mask_batch)
        second_in = self.concat([x_coarse, mask_batch])
        pl1 = self.refinement1(second_in)
        pl2 = self.refinement2(pl1)
        second_out = self.refinement3(pl2)
        second_out = self.refinement4(second_out)
        second_out = self.refinement5(second_out)
        pl3 = self.refinement6(second_out)
        x_hallu = pl3
        x, match = self.contextual_attention(pl3, pl3, mask, self.method)
        x = self.conv_att1(x)
        x = self.cat((x_hallu, x))
        second_out = self.conv_att2(x)
        second_out = self.refinement7(second_out)
        second_out_att = self.apply_attention1(pl2, match)
        second_out = self.concat([second_out_att, second_out])
        second_out = self.refinement8(second_out)
        second_out_att = self.apply_attention2(pl1, match)
        second_out = self.concat([second_out_att, second_out])
        second_out = self.refinement9(second_out)
        second_out = ops.clip_by_value(second_out, -1, 1)
        return first_out, second_out, match


class Discriminator(nn.Cell):
    """
    Build the complete discriminator.

    Return:
        x: The output of discriminator.
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = Conv2dLayer(3, 64, 5, 2, 1)
        self.block2 = Conv2dLayer(64, 128, 5, 2, 1)
        self.block3 = Conv2dLayer(128, 256, 5, 2, 1)
        self.block4 = Conv2dLayer(256, 256, 5, 2, 1)
        self.block5 = Conv2dLayer(256, 256, 5, 2, 1)
        self.block6 = Conv2dLayer(256, 256, 5, 2, 1)
        self.block7 = nn.Dense(16384, 1)

    def construct(self, img):
        """discriminator forward network"""

        x = img
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.reshape([x.shape[0], -1])
        x = self.block7(x)
        return x
