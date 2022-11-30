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
"""Resnet50 backbone."""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore import context

if context.get_context("device_target") == "Ascend":
    ms_cast_type = mstype.float16
else:
    ms_cast_type = mstype.float32


def weight_init_ones(shape):
    """
    Weight init.

    Args:
        shape(List): weights shape.

    Returns:
        Tensor, weights, default float32.
    """
    return Tensor(np.array(np.ones(shape).astype(np.float32) * 0.01).astype(np.float32))


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """
    Conv2D wrapper.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively. Default: 3.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: 1.
        padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
            The data type is an integer or a tuple of four integers. If `padding` is an integer,
            then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: 0.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "pad".

    Outputs:
        Tensor, math '(N, C_{out}, H_{out}, W_{out})' or math '(N, H_{out}, W_{out}, C_{out})'.
    """
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = weight_init_ones(shape)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=False).to_float(mstype.float32)


def _batch_norm2d_init(out_chls, momentum=0.1, affine=True, use_batch_statistics=True):
    """
    Batchnorm2D wrapper.

    Args:
        out_cls (int): The number of channels of the input tensor. Expected input size is (N, C, H, W),
            `C` represents the number of channels
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.1.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        use_batch_statistics (bool):

            - If true, use the mean value and variance value of current batch data and track running mean
              and running variance. Default: True.
            - If false, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to true or false according to the training
              and evaluation mode. During training, the parameter is set to true, and during evaluation, the
              parameter is set to false.
    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:'(N, C_{out}, H_{out}, W_{out})'.
    """
    gamma_init = Tensor(np.array(np.ones(out_chls)).astype(np.float32))
    beta_init = Tensor(np.array(np.ones(out_chls) * 0).astype(np.float32))
    moving_mean_init = Tensor(np.array(np.ones(out_chls) * 0).astype(np.float32))
    moving_var_init = Tensor(np.array(np.ones(out_chls)).astype(np.float32))

    return nn.BatchNorm2d(out_chls, momentum=momentum, affine=affine, gamma_init=gamma_init,
                          beta_init=beta_init, moving_mean_init=moving_mean_init,
                          moving_var_init=moving_var_init,
                          use_batch_statistics=use_batch_statistics)


class ResNetFea(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Tensor): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        weights_update (bool): Weight update flag.

    Inputs:
        - **x** (Tensor) - Input block.

    Outputs:
        Tensor, output block.

    Support Plarforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> ResNetFea(ResidualBlockUsing, [3, 4, 6, 3], [64, 256, 512, 1024], [256, 512, 1024, 2048], False)
    """
    def __init__(self, block, layer_nums, in_channels, out_channels, weights_update=False):
        super(ResNetFea, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        bn_training = False
        self.conv1 = _conv(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = _batch_norm2d_init(64, affine=bn_training, use_batch_statistics=bn_training)
        self.relu = P.ReLU()
        self.maxpool = P.MaxPool(kernel_size=3, strides=2, pad_mode="SAME")
        self.weights_update = weights_update

        if not self.weights_update:
            self.conv1.weight.requires_grad = False

        self.layer1 = self._make_layer(block, layer_nums[0], in_channel=in_channels[0],
                                       out_channel=out_channels[0], stride=1, training=bn_training,
                                       weights_update=self.weights_update)
        self.layer2 = self._make_layer(block, layer_nums[1], in_channel=in_channels[1],
                                       out_channel=out_channels[1], stride=2,
                                       training=bn_training, weights_update=True)
        self.layer3 = self._make_layer(block, layer_nums[2], in_channel=in_channels[2],
                                       out_channel=out_channels[2], stride=2,
                                       training=bn_training, weights_update=True)
        self.layer4 = self._make_layer(block, layer_nums[3], in_channel=in_channels[3],
                                       out_channel=out_channels[3], stride=2,
                                       training=bn_training, weights_update=True)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, training=False, weights_update=False):
        """
        Make layer for resnet backbone.

        Args:
            block (Tensor): ResNet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for convolutional layer.
            training(bool): Whether to do training. Default: False.
            weights_update(bool): Whether to update weights. Default: False.

        Returns:
            SequentialCell, Combine several layers toghter.

        Examples:
            >>> _make_layer(InvertedResidual, 4, 64, 64, 1)
        """
        layers = []
        down_sample = False
        if stride != 1 or in_channel != out_channel:
            down_sample = True
        resblk = block(in_channel, out_channel, stride=stride, down_sample=down_sample,
                       training=training, weights_update=weights_update)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channel, out_channel, stride=1, training=training, weights_update=weights_update)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """Construct ResNet architecture."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        identity = c2
        if not self.weights_update:
            identity = F.stop_gradient(c2)
        c3 = self.layer2(identity)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return identity, c3, c4, c5


class ResidualBlockUsing(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        stride (int): Stride size for the initial convolutional layer. Default: 1.
        down_sample (bool): If to do the downsample in block. Default: False.
        momentum (float): Momentum for batchnorm layer. Default: 0.1.
        training (bool): Training flag. Default: False.
        weights_updata (bool): Weights update flag. Default: False.

    Inputs:
        - **x** (Tensor) - Input block.

    Outputs:
        Tensor, output block.

    Support Plarforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        ResidualBlockUsing(3, 256, stride=2, down_sample=True)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sample=False,
                 momentum=0.1, training=False, weights_update=False):
        super(ResidualBlockUsing, self).__init__()

        self.affine = weights_update

        out_chls = out_channels // self.expansion
        self.conv1 = _conv(in_channels, out_chls, kernel_size=1, stride=1, padding=0)
        self.bn1 = _batch_norm2d_init(out_chls, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        self.conv2 = _conv(out_chls, out_chls, kernel_size=3, stride=stride, padding=1)
        self.bn2 = _batch_norm2d_init(out_chls, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        self.conv3 = _conv(out_chls, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = _batch_norm2d_init(out_channels, momentum=momentum, affine=self.affine,
                                      use_batch_statistics=training)

        if training:
            self.bn1 = self.bn1.set_train()
            self.bn2 = self.bn2.set_train()
            self.bn3 = self.bn3.set_train()

        if not weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False

        self.relu = P.ReLU()
        self.downsample = down_sample
        if self.downsample:
            self.conv_down_sample = _conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.bn_down_sample = _batch_norm2d_init(out_channels, momentum=momentum, affine=self.affine,
                                                     use_batch_statistics=training)
            if training:
                self.bn_down_sample = self.bn_down_sample.set_train()
            if not weights_update:
                self.conv_down_sample.weight.requires_grad = False
        self.add = P.Add()

    def construct(self, x):
        """Construct ResNet V1 residual block."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out
