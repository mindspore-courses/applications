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
# =========================================================================
""" PFLD Backbone network """

from mindspore import ops, nn

from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.classification.models.backbones import InvertedResidual


class PFLDBackbone(nn.Cell):
    """
    PFLD backbone network

    Args:
        channel_num (tuple): With this parameter, to define the input channel and output
            channel of every layer.

    Returns:
        Tensor, Features1 needed for auxiliary network.
        Tensor, Features2 is the entry required for the network head.

    Examples:
        >>> PFLDBackbone(channel_num=(3, 64, 64, 64, 64, 128, 128, 128, 16, 32, 128))
    """

    def __init__(self,
                 channel_num: tuple = (3, 64, 64, 64, 64, 128, 128, 128, 16, 32, 128)):
        super(PFLDBackbone, self).__init__()

        # Input channel, output channel, stride, expansion rate
        self.block1 = ConvNormActivation(channel_num[0], channel_num[1], 3, 2)
        self.block2 = ConvNormActivation(channel_num[1], channel_num[2], 3, 1)

        self.conv3 = InvertedResidual(channel_num[2], channel_num[3], 2, 2)
        self.block3 = self.make_layer(InvertedResidual, 4, channel_num[3], channel_num[4], 1, 2)

        self.conv4 = InvertedResidual(channel_num[4], channel_num[5], 2, 2)
        self.conv5 = nn.SequentialCell(
            ConvNormActivation(channel_num[5], channel_num[5] * 4, 1),
            ConvNormActivation(channel_num[5] * 4, channel_num[5] * 4),
            ConvNormActivation(channel_num[5] * 4, channel_num[6], 1, activation=None))
        self.block5 = self.make_layer(InvertedResidual, 5, channel_num[6], channel_num[7], 1, 4)

        self.conv6 = InvertedResidual(channel_num[7], channel_num[8], 1, 2)
        self.avg_pool1 = nn.AvgPool2d(14)

        self.conv7 = ConvNormActivation(channel_num[8], channel_num[9], 3, 2)
        self.avg_pool2 = nn.AvgPool2d(7)

        self.conv8 = nn.Conv2d(channel_num[9], channel_num[10], 7, 1, pad_mode="pad")
        self.relu = nn.ReLU()

        self.concat_op = ops.Concat(1)

    def make_layer(self,
                   block: nn.Cell,
                   layer_num: int,
                   in_channel: int,
                   out_channel: int,
                   stride: int,
                   expand_ratio: int):
        """
        Make layer for PFLD backbone.

        Args:
            block (Cell): DarkNet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for convolutional layer.
            expand_ratio (int): expand ration of input channel.

        Returns:
            SequentialCell. Combine several layers toghter.

        Examples:
            >>> make_layer(InvertedResidual, 4, 64, 64, 1, 2)
        """

        layers = []
        for _ in range(layer_num):
            pfld_aux_blk = block(in_channel, out_channel, stride, expand_ratio)
            layers.append(pfld_aux_blk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """ build network """
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv3(x)
        features1 = self.block3(x)

        x = self.conv4(features1)
        x = self.conv5(x)
        x = self.block5(x)
        x = self.conv6(x)

        x1 = self.avg_pool1(x)
        x1 = x1.view((x1.shape[0], -1))

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view((x2.shape[0], -1))

        x3 = self.conv8(x)
        x3 = self.relu(x3)
        x3 = x3.view((x3.shape[0], -1))

        multi_scale = self.concat_op((x1, x2, x3))

        return features1, multi_scale