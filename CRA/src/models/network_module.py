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
"""Define network layer module."""

import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal


class Conv2dLayer(nn.Cell):
    """
    Define the convolutional layer in the discriminator, including convolution, activation operations.

    Args:
        in_channels(int): spatial dimension of input tensor.
        out_channels(int): spatial dimension of output tensor.
        kernel_size(int): the height and width of convolution kernel.
        stride(int): the moving step of convolution kernel.
        dilation(int): dilation size of convolution kernel.

    Return:
        x: Output of the network layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(Conv2dLayer, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same', dilation=dilation,
                                has_bias=True, weight_init=TruncatedNormal(0.05))

    def construct(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x


class DepthSeparableConv(nn.Cell):
    """
    Building gate branch of depth-separable LWGC.

    Args:
        in_channel(int): spatial dimension of input tensor.
        out_channel(int): spatial dimension of output tensor.
        stride(int): the moving step of convolution kernel.
        dilation(int): dilation size of convolution kernel.

    Return:
        x: Output of the network layer.
    """

    def __init__(self, in_channel, out_channel, stride, dilation):
        super(DepthSeparableConv, self).__init__()
        self.ds_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride,
                                 pad_mode='same', padding=0, dilation=dilation, group=1, has_bias=True,
                                 weight_init=TruncatedNormal(0.05))

    def construct(self, x):
        x = self.ds_conv(x)
        return x


class ScConv(nn.Cell):
    """
    Building gate branch of single-channel LWGC.

    Args:
        in_channel(int): spatial dimension of input tensor.
        kernel_size(int): the height and width of convolution kernel.
        stride(int): the moving step of convolution kernel.
        padding(int): the number of padding on the height and width directions of the input.
        dilation(int): dilation size of convolution kernel.

    Return:
        x: Output of the network layer.
    """

    def __init__(self, in_channel, kernel_size, stride, padding, dilation):
        super(ScConv, self).__init__()
        self.single_channel_conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=kernel_size,
                                             stride=stride, pad_mode='same', padding=padding, dilation=dilation,
                                             group=1, has_bias=True, weight_init=TruncatedNormal(0.05))

    def construct(self, x):
        x = self.single_channel_conv(x)
        return x


class GatedConv2d(nn.Cell):
    """
    Implement complete depth-separable and single-channel LWGC operation.

    Args:
        in_channel(int): spatial dimension of input tensor.
        out_channel(int): spatial dimension of output tensor.
        kernel_size(int): the height and width of convolution kernel.
        stride(int): the moving step of convolution kernel.
        dilation(int): dilation size of convolution kernel.
        sc(bool): if True, the network is single-channel LWGC; otherwise, it is depth-separable LWGC operation.

    Return:
        x: Output of the network layer.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, sc=False):
        super(GatedConv2d, self).__init__()
        self.activation = nn.ELU(alpha=1.0)
        if sc:
            self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode='same', padding=0,
                                    dilation=dilation, has_bias=True, weight_init=TruncatedNormal(0.05))
            self.gate_factor = ScConv(in_channel, kernel_size, stride, 0, dilation)
        else:
            self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode='same', padding=0,
                                    dilation=dilation, has_bias=True, weight_init=TruncatedNormal(0.05))
            self.gate_factor = DepthSeparableConv(in_channel, out_channel, stride, dilation)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        gc_f = self.conv2d(x)
        gc_g = self.gate_factor(x)
        x = self.sigmoid(gc_g) * self.activation(gc_f)
        return x


class TransposeGatedConv2d(nn.Cell):
    """
    Add upsampling operation to gated convolution.

    Args:
        in_channel(int): spatial dimension of input tensor.
        out_channel(int): spatial dimension of output tensor.
        kernel_size(int): the height and width of convolution kernel.
        stride(int): the moving step of convolution kernel.
        dilation(int): dilation size of convolution kernel.
        sc(bool): if True, the network is single-channel LWGC; otherwise, it is depth-separable LWGC operation.
        scale_factor(int): the scale factor of new size of the tensor.

    Return:
        x: Output of the network layer.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation=1, sc=False, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.gate_conv2d = GatedConv2d(in_channel, out_channel, kernel_size, stride, dilation, sc)

    def construct(self, x):
        x = nn.ResizeBilinear()(x, scale_factor=self.scale_factor)
        x = self.gate_conv2d(x)
        return x
