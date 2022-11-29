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
"""ReCoNet model."""
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net, Parameter
from mindspore.common import initializer as init
from mindspore import ops, Tensor


class InstanceNorm2d(nn.Cell):
    """InstanceNorm2d copy from model zoo"""

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros'):
        super().__init__()
        self.num_features = num_features
        self.moving_mean = Parameter(init.initializer('zeros', num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(init.initializer('ones', num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(init.initializer(gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(init.initializer(beta_init, num_features), name="beta", requires_grad=affine)
        self.sqrt = ops.Sqrt()
        self.eps = Tensor(np.array([eps]), mindspore.float32)
        self.cast = ops.Cast()

    def construct(self, x):
        """calculate InstanceNorm output"""
        mean = ops.ReduceMean(keep_dims=True)(x, (2, 3))
        mean = self.cast(mean, mindspore.float32)
        tmp = x - mean
        tmp = tmp * tmp
        var = ops.ReduceMean(keep_dims=True)(tmp, (2, 3))
        std = self.sqrt(var + self.eps)
        gamma_t = self.cast(self.gamma, mindspore.float32)
        beta_t = self.cast(self.beta, mindspore.float32)
        x = (x - mean) / std * gamma_t.reshape(1, -1, 1, 1) + beta_t.reshape(1, -1, 1, 1)
        return x


class ConvLayer(nn.Cell):
    """
    Conv2d layer

    Args:
        in_channels (int): In channel size
        out_channels (int): Out channel size
        kernel_size (int): Kernel size
        stride (int): Stride

    Returns:
        Tensor, conv result
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                has_bias=True,
                                pad_mode='valid')

    def construct(self, x):
        """Construct ConvLayer."""
        x = mindspore.numpy.pad(x,
                                (
                                    (0, 0),
                                    (0, 0),
                                    (self.kernel_size // 2, self.kernel_size // 2),
                                    (self.kernel_size // 2, self.kernel_size // 2)
                                ),
                                mode='reflect')
        x = self.conv2d(x)
        return x


class ConvNormLayer(nn.Cell):
    """
    Conv2d with InstanceNorm

    Args:
        in_channels (int): In channel size
        out_channels (int): Out channel size
        kernel_size (int): Kernel size
        stride (int): Stride
        activation (bool): Whether conv layer have activation function

    Returns:
        Tensor, conv result
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=True):
        super().__init__()
        layers = [
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            InstanceNorm2d(out_channels, affine=True)
        ]
        if activation:
            layers.append(nn.ReLU())

        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Construct ConvNormLayer."""
        x = self.layers(x)
        return x


class ResLayer(nn.Cell):
    """
    ReCoNet res layer

    Args:
        in_channels (int): In channel size
        out_channels (int): Out channel size
        kernel_size (int): Kernel size

    Returns:
        Tensor, res layer result
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.branch = nn.SequentialCell(
            [
                ConvNormLayer(in_channels, out_channels, kernel_size, 1),
                ConvNormLayer(out_channels, out_channels, kernel_size, 1, activation=False)
            ]
        )

        self.activation = nn.ReLU()

    def construct(self, x):
        """Construct ResLayer."""
        x = x + self.branch(x)
        x = self.activation(x)
        return x


class ConvTanhLayer(nn.Cell):
    """
    Conv2d with tanh activation function

    Args:
        in_channels (int): In channel size
        out_channels (int): Out channel size
        kernel_size (int): Kernel size
        stride (int): Stride

    Returns:
        Tensor, res layer result
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.SequentialCell(
            [
                ConvLayer(in_channels, out_channels, kernel_size, stride),
                nn.Tanh()
            ]
        )

    def construct(self, x):
        """Construct ConvTanhLayer."""
        x = self.layers(x)
        return x


class Encoder(nn.Cell):
    """
    ReCoNet encoder layer

    Returns:
        Tensor, encoder result
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.SequentialCell(
            [
                ConvNormLayer(3, 48, 9, 1),
                ConvNormLayer(48, 96, 3, 2),
                ConvNormLayer(96, 192, 3, 2),
                ResLayer(192, 192, 3),
                ResLayer(192, 192, 3),
                ResLayer(192, 192, 3),
                ResLayer(192, 192, 3)
            ]
        )

    def construct(self, x):
        """Construct Encoder."""
        x = self.layers(x)
        return x


class Decoder(nn.Cell):
    """
    ReCoNet decoder layer

    Returns:
        Tensor, decoder result
    """

    def __init__(self):
        super().__init__()
        self.up_sample = nn.ResizeBilinear()
        self.conv1 = ConvNormLayer(192, 96, 3, 1)
        self.conv2 = ConvNormLayer(96, 48, 3, 1)
        self.conv3 = ConvTanhLayer(48, 3, 9, 1)

    def construct(self, x):
        """Construct Decoder."""
        x = self.up_sample(x, scale_factor=2)
        x = self.conv1(x)
        x = self.up_sample(x, scale_factor=2)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ReCoNet(nn.Cell):
    """
    ReCoNet model

    Returns:
        Tensor, ReCoNet result
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def construct(self, x):
        """Construct ReCoNet."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_reconet(ckpt_file: str = None):
    """
    Load ReCoNet from ckpt

    Args:

        ckpt_file (str): The path of checkpoint files. Default: None.

    Outputs:
        ReCoNet Model
    """
    model = ReCoNet()

    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(model, param_dict)

    return model
