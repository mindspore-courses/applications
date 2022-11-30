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
"""vogue_block"""

import numpy as np
from mindspore import nn, ops, Tensor, Parameter
import mindspore as ms

from vogue_utils import conv2d_resample, bias_act


# Captured from the checkpoint
resample_filter = Tensor([[0.0156, 0.0469, 0.0469, 0.0156],
                          [0.0469, 0.1406, 0.1406, 0.0469],
                          [0.0469, 0.1406, 0.1406, 0.0469],
                          [0.0156, 0.0469, 0.0469, 0.0156]], ms.float32)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    """
    Normalize 2nd moment.

    Args:
        x (Tensor): Input tensor.
        dim (int): Axis. Default: 1.
        eps (float): Small value added to the denominator. Default: 1e-8.

    Returns:
        Tensor, output tensor of normalization.

    Examples:
        >>> x = normalize_2nd_moment(x)
    """
    square = ops.Square()
    sqrt = ops.Sqrt()
    return x / sqrt(square(x).mean(axis=dim, keep_dims=True) + eps)


class FullyConnectedLayer(nn.Cell):
    """
    Fully Connected Layer.

    Args:
        in_features (int): Number of input features.
        out_features(int): Number of output features.
        bias (bool): Apply additive bias before the activation function? Default: True.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'linear'.
        lr_multiplier (float): Learning rate multiplier. Default: 1.0.
        bias_init (int): Initial value for the additive bias. Default: 0.

    Inputs:
        - **x** (Tensor) - Input tensor.

    Outputs:
        Tensor, fully connected output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = FullyConnectedLayer(in_features, out_features, lr_multiplier=lr_multiplier)
        >>> x = layer(x)
    """
    def __init__(self, in_features, out_features, bias=True, activation='linear', lr_multiplier=1.0, bias_init=0):
        super(FullyConnectedLayer, self).__init__()
        self.activation = activation
        np.random.seed(0)
        self.weight = Parameter(Tensor(np.random.randn(out_features, in_features) / lr_multiplier, ms.float32))
        self.bias = Parameter(Tensor(np.full([out_features], np.float32(bias_init)), ms.float32)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def construct(self, x):
        """Fully_connected_layer construct"""
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = b.expand_dims(axis=0) + ops.matmul(x, w.transpose())
        else:
            x = ops.matmul(x, w.transpose())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class Conv2dLayer(nn.Cell):
    """
    Conv2d Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels(int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        bias (bool): Apply additive bias before the activation function? Default: True.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'linear'.
        up (int): Integer upsampling factor. Default: 1.
        down (int): Integer downsampling factor. Default: 1.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        trainable (bool): Update the weights of this layer during training? Default: True.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **gain** (int) - Gain on act_gain. Default: 1.
        - **all_info** (list) - Information of all_conv. Default: None.

    Outputs:
        Tensor, 2d convolution output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = Conv2dLayer(in_channels, out_channels, kernel_size, channels_last)
        >>> x = layer(x, all_info=all_info)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, activation='linear',
                 up=1, down=1, conv_clamp=None, trainable=True):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation]['def_gain']

        zeros = ops.Zeros()
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), ms.float32)
        bias = zeros(out_channels, ms.float32) if bias else None
        if trainable:
            self.weight = Parameter(weight)
            self.bias = Parameter(bias) if bias is not None else None
        else:
            self.weight = weight
            if bias is not None:
                self.bias = bias
            else:
                self.bias = None

    def construct(self, x, gain=1, all_info=None):
        """Conv2d construct"""
        w = self.weight * self.weight_gain
        b = self.bias if self.bias is not None else None
        flip_weight = (self.up == 1)
        x = conv2d_resample.conv2d_resample(x=x, w=w, f=resample_filter, up=self.up,
                                            down=self.down, padding=self.padding, flip_weight=flip_weight,
                                            all_info=all_info)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class MappingNetwork(nn.Cell):
    """
    Mapping Network.

    Args:
        z_dim (int): Input latent (Z) dimensionality, 0 = no latent.
        c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
        w_dim (int): Intermediate latent (W) dimensionality.
        num_ws (int): Number of intermediate latents to output, None = do not broadcast.
        num_layers (int): Number of mapping layers. Default: 8.
        embed_features (bool): Label embedding dimensionality. Default: None.
        layer_features (bool): Number of intermediate features in the mapping layers,
            None = same as w_dim. Default: None.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'lrelu'.
        lr_multiplier (float): Learning rate multiplier for the mapping layers. Default: 0.01.
        w_avg_beta (float): Decay for tracking the moving average of W during training,
            None = do not track. Default: 0.995.

    Inputs:
        - **z** (Tensor) - Latent tensor.
        - **c** (Tensor) - Label tensor.
        - **truncation_psi** (int) - Truncation coefficient. Default: 1.
        - **truncation_cutoff** (int) - Truncation cutoff if truncation_psi != 1. Default: None.
        - **skip_w_avg_update** (bool) - Need to skip the w_avg undate. Default: False.

    Outputs:
        Tensor, mapping network output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> mapping = MappingNetwork(z_dim, c_dim, w_dim, num_ws)
        >>> ws = mapping(z, c)
    """
    def __init__(self, z_dim, c_dim, w_dim, num_ws, num_layers=8, embed_features=None, layer_features=None,
                 activation='lrelu', lr_multiplier=0.01, w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        zeros = ops.Zeros()

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        self.blocks = nn.CellList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            self.blocks.append(layer)

        if num_ws is not None and w_avg_beta is not None:
            self.w_avg = Parameter(zeros(w_dim, ms.float32))

    def construct(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        """Mapping network construct"""
        concat = ops.Concat(axis=1)
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z)
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c))
            x = concat((x, y)) if x is not None else y

        for layer in self.blocks:
            x = layer(x)

        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg = (x.mean(axis=0) + (self.w_avg - x.mean(axis=0)) * self.w_avg_beta).copy()

        if self.num_ws is not None:
            x = x.expand_dims(axis=1).repeat(self.num_ws, axis=1)

        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg + (x - self.w_avg) * truncation_psi
            else:
                x[:, :truncation_cutoff] = self.w_avg + (x[:, :truncation_cutoff] - self.w_avg) * truncation_psi
        return x
