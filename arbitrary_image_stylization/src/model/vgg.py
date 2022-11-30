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
""" Implementation of VGG-16 network for deriving content and style loss. """
from mindspore import nn, Tensor

class VGG(nn.Cell):
    """
    VGG-16 implementation intended for representation of content and style.
    Specially, replace max-pooling with average-pooling and use conv2d indstead of fully connected layers.

    Args:
        in_channel: number of input channels. Default: 3

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        A dict mapping layer names to their corresponding Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> encoder = VGG()
        >>> x = Tensor(np.ones([1, 10, 128, 128]), mindspore.float32)
        >>> output = encoder(x)
    """
    def __init__(self, in_channel=3):
        super(VGG, self).__init__()
        self.conv1 = self.make_layer(2, in_channel, 64, 3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = self.make_layer(2, 64, 128, 3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = self.make_layer(3, 128, 256, 3)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = self.make_layer(3, 256, 512, 3)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = self.make_layer(3, 512, 512, 3)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv6 = BasicConv2d(512, 4096, kernel_size=7, pad_mode='valid')
        self.dropout1 = nn.Dropout(0.5)
        self.conv7 = BasicConv2d(4096, 4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.5)
        self.conv8 = BasicConv2d(4096, 1000, kernel_size=1, activation_fn=None)

    def make_layer(self, repeat, in_channel, out_channel, kernel_size):
        layer = []
        for _ in range(repeat):
            layer.append(BasicConv2d(in_channel, out_channel, kernel_size=kernel_size))
            in_channel = out_channel
        return nn.SequentialCell(layer)

    def construct(self, x):
        """ forward process """
        x *= 255.0
        _, _, height, width = x.shape
        cons = Tensor([123.68, 116.779, 103.939]).expand_dims(1).expand_dims(1)
        cons = cons.repeat(height, 1).repeat(width, 2).expand_dims(0)
        x -= cons

        end_points = {}
        x = self.conv1(x)
        end_points['vgg_16/conv1'] = x

        x = self.pool1(x)
        x = self.conv2(x)
        end_points['vgg_16/conv2'] = x

        x = self.pool2(x)
        x = self.conv3(x)
        end_points['vgg_16/conv3'] = x

        x = self.pool3(x)
        x = self.conv4(x)
        end_points['vgg_16/conv4'] = x

        x = self.pool4(x)
        x = self.conv5(x)
        end_points['vgg_16/conv5'] = x

        x = self.pool5(x)
        x = self.conv6(x)
        end_points['vgg_16/conv6'] = x

        x = self.dropout1(x)
        x = self.conv7(x)
        end_points['vgg_16/conv7'] = x

        x = self.dropout2(x)
        x = self.conv8(x)
        end_points['vgg_16/fc8'] = x

        return end_points

class BasicConv2d(nn.Cell):
    """
    Convolution operation followed by an activation function.

    Args:
        in_channels (int): Input channels of convolution.
        out_channels (int): Output channels of convolution.
        kernel_size (int): Shape of convolution kernel, expected to be odd. Default: 3.
        stride (int): Stride of convolution. Default: 1.
        activation_fn (nn.Cell): Activation operation after convolution. Default: nn.ReLU().
        pad_mode (str): Padding mode of convolution. Optional values are "same", "valid", "pad", Default: "same".

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> conv = BasicConv2d(in_channels=3, out_channels=32, kernel_size=3)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1,
                 activation_fn=nn.ReLU(), pad_mode='same', **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, has_bias=True, **kwargs)
        self.activation_fn = activation_fn

    def construct(self, x):
        x = self.conv(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x
