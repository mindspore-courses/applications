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
# ======================================================================
"""resnet"""
import mindspore.nn as nn
from mindspore import ops


__all__ = ['resnet50', 'resnet101']


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, pad_mode='pad')


def _conv3x3(in_channel, out_channel, stride=1, is_dilation=False):
    if is_dilation:
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=2, pad_mode='pad', dilation=2)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, pad_mode='pad')


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=3, pad_mode='pad')


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.9)


class ResidualBlock(nn.Cell):
    """
    ResNet50 and ResNet101 residual block definition"

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int, optional): Stride size for the first convolutional layer. Defaults to 1.
        is_dilation (bool, optional): Use dilated Convolution. Defaults to False.

    Returns:
        Cell, cell instance of residual block.

    Inputs:
        - **x** (Tensor) - The input tensor. The shape can be [batch, dim, H, W].

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, is_dilation=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion

        self.conv1 = _conv1x1(in_channel, channel, stride=1)  # 1x1 convolution
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride, is_dilation=is_dilation)  # 3x3 convolution
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)  # 1x1 convolution
        self.bn3 = _bn(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:  # Downsampling
            self.down_sample = True
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

    def construct(self, x):
        """get block output"""
        identity = x

        out = self.conv1(x)  # 1x1 convolution
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 3x3 convolution
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 1x1 convolution
        out = self.bn3(out)

        if self.down_sample:  # Downsampling
            identity = self.down_sample_layer(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        return_interm_layers (bool): Get the result of the middle layer. Default: False.
        is_dilation (bool): Use dilated Convolution. Default: False.

    Returns:
        Cell, cell instance of residual block.

    Inputs:
        - **x** (Tensor) - The input tensor. The shape can be [batch, 3, H, W].

    Outputs:
        list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2])
    """

    def __init__(self, block, layer_nums, in_channels, out_channels, strides,
                 return_interm_layers=False, is_dilation=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.return_interm_layers = return_interm_layers
        # Layer 1 7x7 convolution.
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = ops.ReLU()
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        # The first subnet. Input 64. Output 25.
        self.layer1 = self._make_layer(block, layer_nums[0], in_channel=in_channels[0],
                                       out_channel=out_channels[0], stride=strides[0])
        # The second subnet Layer. Input 256. Output 512.
        self.layer2 = self._make_layer(block, layer_nums[1], in_channel=in_channels[1],
                                       out_channel=out_channels[1], stride=strides[1])
        # The third subnet layer. Input 512. Output 1024.
        self.layer3 = self._make_layer(block, layer_nums[2], in_channel=in_channels[2],
                                       out_channel=out_channels[2], stride=strides[2])
        # The 4th subnet Layer. Input 1024 Output 2048.
        if is_dilation:
            strides[3] = 1
        self.layer4 = self._make_layer(block, layer_nums[3], in_channel=in_channels[3],
                                       out_channel=out_channels[3], stride=strides[3], is_dilation=is_dilation)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, is_dilation=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            is_dilation (bool): Use dilated Convolution. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1, is_dilation=is_dilation)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """ Get resnet output """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)  # 2-10
        c3 = self.layer2(c2)  # 11-22
        c4 = self.layer3(c3)  # 23-40
        c5 = self.layer4(c4)  # 41-49

        if self.return_interm_layers:
            return [c2, c3, c4, c5]

        return [c5]


def resnet50(return_interm_layers=False, is_dilation=False):
    """
    Get ResNet50 neural network.

    Args:
        return_interm_layers (bool): Get the result of the middle layer. Default: False.
        is_dilation (bool): Use dilated Convolution. Default: False.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50()
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  return_interm_layers=return_interm_layers,
                  is_dilation=is_dilation)


def resnet101(return_interm_layers=False, is_dilation=False):
    """
    Get ResNet01 neural network.

    Args:
        return_interm_layers (bool): Get the result of the middle layer. Default: False.
        is_dilation (bool): Use dilated Convolution. Default: False.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet101()
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  return_interm_layers=return_interm_layers,
                  is_dilation=is_dilation)
