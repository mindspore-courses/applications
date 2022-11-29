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
"""ResNet50 with multiple layers of output."""

import numpy as np
from mindspore import nn, ops, Tensor
import mindspore


def conv(kernel_size, in_channel, out_channel, stride):
    """
    Convolution with particular kernel size.

    Args:
        kernel_size(int): The size of convolution kernel.
        in_channel(int): The number of input channel.
        out_channel(int): The number of output channel.
        stride(int): The stride number of convolution operation. Default: 1.

    Returns:
        A nn.Cell, represents the convolution operation.
    """
    weight_shape = (out_channel, in_channel, kernel_size, kernel_size)
    weight = _weight_variable(weight_shape)
    pad = (kernel_size - 1) // 2
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=kernel_size, stride=stride, padding=pad, pad_mode='pad', weight_init=weight)


def _fc(in_channel, out_channel):
    """
    Construct a fully connected layer with in_channel as input channel size, out_channel as output channel size.

    Args:
        in_channel(int): The number of input channel.
        out_channel(int): The number of output channel.

    Returns:
        A nn.Cell, represents the dense layer.
    """
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    """
    Residual block of resnet.

    Args:
        in_channel(int): Channel number of input tensor.
        out_channel(int): Channel number of output tensor.
        stride(int): The stride number of 3 by 3 convolution layer. Default: 1.

    Inputs:
        - **x** (Tensor) - An input tensor, which shape is :math:`(B, in_channel, H, W)`.

    Outputs:
        Tensor of the output of the feature pyramid, which shape is :math:`(B, out_channel, H, W)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from src.model.resnet50 import ResidualBlock
        >>> from mindspore import ops
        >>> mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
        >>> block = ResidualBlock(4, 4)
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 4, 64, 64), mindspore.float32)
        >>> print(block(x).shape)
        (1, 4, 64, 64)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = conv(1, in_channel, channel, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = conv(3, channel, channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = conv(1, channel, out_channel, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([conv(1, in_channel, out_channel, stride),
                                                        nn.BatchNorm2d(out_channel)])
        self.add = ops.Add()

    def construct(self, x):
        """Forward pass."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet structure.

    Args:
        block(nn.Cell): The block for ResNet construction.
        layer_nums(list): The layer number of different stages.
        in_channels(list): The input channel number of different stages.
        out_channels(list): The output channel number of different stages.
        strides(list): The stride number of different stages.

    Inputs:
        - **x** (Tensor) - An input tensor, which shape is :math:`(B, in_channel, H, W)`.

    Outputs:
        Tuple of 3 Tensor, which contains output Tensor from top, middle and bottom layer.

        - **tensor1** (Tensor) - Input tensor of the top layer, which shape is :math:`(B, out_channels[1], H1, W1)`.
        - **tensor2** (Tensor) - Input tensor of the medium layer, which shape is :math:`(B, out_channels[2], H2, W2)`.
        - **tensor3** (Tensor) - Input tensor of the bottom layer, which shape is :math:`(B, out_channels[3], H3, W3)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from src.model.resnet50 import ResNet, ResidualBlock
        >>> from mindspore import ops
        >>> mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
        >>> resnet = ResNet(ResidualBlock,
        >>>                 [3, 4, 6, 3],
        >>>                 [64, 256, 512, 1024],
        >>>                 [256, 512, 1024, 2048],
        >>>                 [1, 2, 2, 2])
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 3, 128, 128), mindspore.float32)
        >>> result = resnet(x)
        >>> print(result[0].shape, result[1].shape, result[2].shape)
        (1, 512, 16, 16) (1, 1024, 8, 8) (1, 2048, 4, 4)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = conv(7, 3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = ops.ReLU()

        self.zeros1 = ops.Zeros()
        self.zeros2 = ops.Zeros()
        self.concat1 = ops.Concat(axis=2)
        self.concat2 = ops.Concat(axis=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Construct a ResNet stage layer.

        Args:
            block(nn.Cell): The block for ResNet construction.
            layer_num(int): The layer number of the stage.
            in_channel(int): The input channel number of the stage.
            out_channel(int): The output channel number of the stage.
            stride(int): The stride number of the stage.

        Returns:
            A nn.Cell, represents the stage layer of ResNet.
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        zeros1 = self.zeros1((x.shape[0], x.shape[1], 1, x.shape[3]), mindspore.float32)
        x = self.concat1((zeros1, x))
        zeros2 = self.zeros2((x.shape[0], x.shape[1], x.shape[2], 1), mindspore.float32)
        x = self.concat2((zeros2, x))

        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


def resnet50():
    """Construct ResNet50 with 3 output stages."""
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2])


def _weight_variable(shape, factor=0.01):
    """Use standard normal distribution to initialize tensor."""
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)
