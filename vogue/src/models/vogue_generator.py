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
"""vogue_generator_block"""

import numpy as np
from mindspore import nn, ops, Tensor, Parameter
import mindspore as ms

from vogue_utils import conv2d_resample, upfirdn2d, bias_act
from models.pose_encoder import PoseEncoder
from models.vogue_block import FullyConnectedLayer, MappingNetwork, resample_filter


def modulated_conv2d(x, weight, styles, noise=None, up=1, down=1, padding=0, r_filter=None,
                     demodulate=True, flip_weight=True, fused_modconv=True, all_info=None):
    """
    Modulated conv2d.

    Args:
        x (Tensor): Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight (Tensor): Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles (Tensor): Modulation coefficients of shape [batch_size, in_channels].
        noise (Tensor): Optional noise tensor to add to the output activations. Default: None.
        up (int): Integer upsampling factor. Default: 1.
        down (int): Integer downsampling factor. Default: 1.
        padding (int): Padding with respect to the upsampled image. Default: 0.
        r_filter (Tensor): Low-pass filter to apply when resampling activations. Default: None.
        demodulate (bool): Apply weight demodulation?  False = convolution, True = correlation. Default: True.
        flip_weight (bool): Need to flip the weight? Default: True.
        fused_modconv (bool): Perform modulation, convolution, and demodulation as a single fused operation?
           . Default: True.
        all_info (list): Information of all_conv. Default: None.

    Returns:
        Tensor, modulated output tensor.

    Examples:
        >>> x = modulated_conv2d(x, weight, styles, all_info=all_info)
    """
    square = ops.Square()
    sqrt = ops.Sqrt()
    batch_size = x.shape[0]
    _, in_channels, kh, kw = weight.shape

    if x.dtype == ms.float16 and demodulate:
        weight_norm = weight.max(axis=[1, 2, 3], keepdims=True)
        styles_norm = styles.max(axis=1, keepdims=True)
        weight_shape = in_channels * kh * kw
        weight_shape_tensor = Tensor(weight_shape, ms.float32)
        weight_sqrt = sqrt(weight_shape_tensor)
        weight_multiplier = 1 / weight_sqrt
        weight = weight * weight_multiplier
        weight = weight / weight_norm
        styles = styles / styles_norm
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.expand_dims(axis=0)
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)

    if demodulate:
        dcoefs = 1 / sqrt(square(w).sum(axis=2).sum(axis=2).sum(axis=2) + 1e-8)
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)

    if not fused_modconv:
        x = x * styles.astype(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.astype(x.dtype), f=r_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight, all_info=all_info)
        if demodulate and noise is not None:
            x = x * dcoefs.astype(x.dtype).reshape(batch_size, - 1, 1, 1) + noise.astype(x.dtype)
        elif demodulate:
            x = x * dcoefs.astype(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x + noise.astype(x.dtype)
        return x

    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.astype(x.dtype), f=r_filter, up=up, down=down,
                                        padding=padding, groups=batch_size, flip_weight=flip_weight, all_info=all_info)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x + noise
    return x


class SynthesisLayer(nn.Cell):
    """
    Synthesis Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        resolution (int): Resolution of this layer.
        kernel_size (int): Convolution kernel size. Default: 3.
        up (int): Integer upsampling factor. Default: 1.
        use_noise (bool): Enable noise input? Default: True.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'lrelu'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **w** (Tensor) - MModulation tensor.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.
        - **fused_modconv** (bool) - Perform modulation, convolution, and demodulation as a single fused operation?
            Default: True.
        - **gain** (int) - Gain on act_gain. Default: 1.
        - **all_info** (list) - Information of all_conv. Default: None.

    Outputs:
        Tensor, synthesis layer output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = SynthesisLayer(in_channels, out_channels, w_dim, resolution, conv_clamp)
        >>> x = layer(x, w, all_info=all_info)
    """
    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, up=1, use_noise=True,
                 activation='lrelu', conv_clamp=None):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation]['def_gain']

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = Parameter(Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size),
                                       ms.float32))
        zeros = ops.Zeros()
        if use_noise:
            self.noise_const = Tensor(np.random.randn(resolution, resolution), ms.float32)
            self.noise_strength = Parameter(zeros((), ms.float32))
        self.bias = Parameter(zeros(out_channels, ms.float32))

    def construct(self, x, w, noise_mode=0, fused_modconv=True, gain=1, all_info=None):
        """Synthesis layer construct"""
        styles = self.affine(w)
        # 0: const, 1: random
        noise = None
        if self.use_noise and noise_mode == 1:
            noise = Tensor(np.random.randn(x.shape[0], 1, self.resolution, self.resolution), ms.float32)\
                    * self.noise_strength
        if self.use_noise and noise_mode == 0:
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, padding=self.padding,
                             r_filter=resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv,
                             all_info=all_info)
        act_gain = self.act_gain * gain
        if self.conv_clamp is not None:
            act_clamp = self.conv_clamp * gain
        else:
            act_clamp = None
        x = bias_act.bias_act(x, self.bias.astype(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class ToRGBLayer(nn.Cell):
    """
    To RGB Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        kernel_size (int): Convolution kernel size. Default: 1.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **w** (Tensor) - MModulation tensor.
        - **fused_modconv** (bool) - Perform modulation, convolution, and demodulation as a single fused operation?
            Default: True.
        - **all_info** (list) - Information of all_conv. Default: None.

    Outputs:
        Tensor, to-rgb layer output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = SynthesisLayer(in_channels, out_channels, w_dim, kernel_size, conv_clamp)
        >>> x = layer(x, w, all_info=all_info)
    """
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None):
        super().__init__()
        zeros = ops.Zeros()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = Parameter(Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size),
                                       ms.float32))
        self.bias = Parameter(zeros(out_channels, ms.float32))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def construct(self, x, w, fused_modconv=True, all_info=None):
        """To rgb layer construct"""
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False,
                             fused_modconv=fused_modconv, all_info=all_info)
        x = bias_act.bias_act(x, self.bias.astype(x.dtype), clamp=self.conv_clamp)
        return x


class SynthesisBlock(nn.Cell):
    """
    Synthesis Block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        resolution (int): Resolution of this layer.
        img_channels (int): Number of output color channels.
        is_last (bool): Is this the last block?
        architecture (str): Architecture: 'orig', 'skip'. Default: 'skip'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        use_fp16 (bool): Use FP16 for this block? Default: False.
        batch_size (int): Batch size. Default: 1.
        train (bool): True: train, False: infer. Default: False.
        layer_kwargs (dict): Arguments for SynthesisLayer.

    Inputs:
        - **x** (Tensor) - Input feature.
        - **img** (Tensor) - Input image.
        - **ws** (Tensor) - Intermediate latents.
        - **force_fp32** (bool) - If force the input to float32. Default: False.
        - **fused_modconv** (bool) - Perform modulation, convolution, and demodulation as a single fused operation?
           . Default: True.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.

    Outputs:
        Tensor, output feature.
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> block = SynthesisBlock(in_channels, out_channels, w_dim, resolution, img_channels, is_last)
        >>> x, img = block(x, img, ws)
    """
    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, is_last, architecture='skip',
                 conv_clamp=None, use_fp16=False, batch_size=1, train=False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.size = batch_size
        self.train = train
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        conv_clamp=conv_clamp, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp)
            self.num_torgb += 1

        self.name_list = ['conv2d'] * 2 + (['transpose2d'] + ['conv2d'] * 4) * 6
        self.dtype_list = ['float32' for _ in range(len(self.name_list))]

        if self.train:
            self.input_list = [(self.size, 512, 4, 4), (self.size, 512, 4, 4), (self.size, 512, 4, 4),
                               (self.size, 512, 11, 11), (self.size, 512, 8, 8), (self.size, 3, 11, 11),
                               (self.size, 512, 8, 8), (self.size, 512, 8, 8), (self.size, 512, 19, 19),
                               (self.size, 512, 16, 16), (self.size, 3, 19, 19), (self.size, 512, 16, 16),
                               (self.size, 512, 16, 16), (self.size, 512, 35, 35), (self.size, 512, 32, 32),
                               (self.size, 3, 35, 35), (self.size, 512, 32, 32), (self.size, 512, 32, 32),
                               (self.size, 256, 67, 67), (self.size, 256, 64, 64), (self.size, 3, 67, 67),
                               (self.size, 256, 64, 64), (self.size, 256, 64, 64), (self.size, 128, 131, 131),
                               (self.size, 128, 128, 128), (self.size, 3, 131, 131), (self.size, 128, 128, 128),
                               (self.size, 128, 128, 128), (self.size, 64, 259, 259), (self.size, 64, 256, 256),
                               (self.size, 3, 259, 259), (self.size, 64, 256, 256)]

            self.weight_list = [(512, 512, 3, 3), (3, 512, 1, 1)] + \
                               [(512, 512, 3, 3), (512, 1, 4, 4), (512, 512, 3, 3), (3, 1, 4, 4), (3, 512, 1, 1)] * 3 \
                               + [(512, 256, 3, 3), (256, 1, 4, 4), (256, 256, 3, 3), (3, 1, 4, 4), (3, 256, 1, 1),
                                  (256, 128, 3, 3), (128, 1, 4, 4), (128, 128, 3, 3), (3, 1, 4, 4), (3, 128, 1, 1),
                                  (128, 64, 3, 3), (64, 1, 4, 4), (64, 64, 3, 3), (3, 1, 4, 4), (3, 64, 1, 1)]

            self.all_conv = nn.CellList([self._conv2d(512, 512, 3, 1, 1, 1, 1, 0),
                                         self._conv2d(512, 3, 1, 1, 0, 1, 1, 1),
                                         self._transpose2d(512, 512, 3, 2, 0, 1, 1, 2),
                                         self._conv2d(512, 512, 4, 1, 0, 1, 512, 3),
                                         self._conv2d(512, 512, 3, 1, 1, 1, 1, 4),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 5),
                                         self._conv2d(512, 3, 1, 1, 0, 1, 1, 6),
                                         self._transpose2d(512, 512, 3, 2, 0, 1, 1, 7),
                                         self._conv2d(512, 512, 4, 1, 0, 1, 512, 8),
                                         self._conv2d(512, 512, 3, 1, 1, 1, 1, 9),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 10),
                                         self._conv2d(512, 3, 1, 1, 0, 1, 1, 11),
                                         self._transpose2d(512, 512, 3, 2, 0, 1, 1, 12),
                                         self._conv2d(512, 512, 4, 1, 0, 1, 512, 13),
                                         self._conv2d(512, 512, 3, 1, 1, 1, 1, 14),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 15),
                                         self._conv2d(512, 3, 1, 1, 0, 1, 1, 16),
                                         self._transpose2d(512, 256, 3, 2, 0, 1, 1, 17),
                                         self._conv2d(256, 256, 4, 1, 0, 1, 256, 18),
                                         self._conv2d(256, 256, 3, 1, 1, 1, 1, 19),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 20),
                                         self._conv2d(256, 3, 1, 1, 0, 1, 1, 21),
                                         self._transpose2d(256, 128, 3, 2, 0, 1, 1, 22),
                                         self._conv2d(128, 128, 4, 1, 0, 1, 128, 23),
                                         self._conv2d(128, 128, 3, 1, 1, 1, 1, 24),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 25),
                                         self._conv2d(128, 3, 1, 1, 0, 1, 1, 26),
                                         self._transpose2d(128, 64, 3, 2, 0, 1, 1, 27),
                                         self._conv2d(64, 64, 4, 1, 0, 1, 64, 28),
                                         self._conv2d(64, 64, 3, 1, 1, 1, 1, 29),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 30),
                                         self._conv2d(64, 3, 1, 1, 0, 1, 1, 31)])
        else:
            self.input_list = [(1, 512*self.size, 4, 4), (1, 512*self.size, 4, 4), (1, 512*self.size, 4, 4),
                               (1, 512*self.size, 11, 11), (1, 512*self.size, 8, 8), (self.size, 3, 11, 11),
                               (1, 512*self.size, 8, 8), (1, 512*self.size, 8, 8), (1, 512*self.size, 19, 19),
                               (1, 512*self.size, 16, 16), (self.size, 3, 19, 19), (1, 512*self.size, 16, 16),
                               (self.size, 512, 16, 16), (self.size, 512, 35, 35), (self.size, 512, 32, 32),
                               (self.size, 3, 35, 35), (self.size, 512, 32, 32), (self.size, 512, 32, 32),
                               (self.size, 256, 67, 67), (self.size, 256, 64, 64), (self.size, 3, 67, 67),
                               (self.size, 256, 64, 64), (self.size, 256, 64, 64), (self.size, 128, 131, 131),
                               (self.size, 128, 128, 128), (self.size, 3, 131, 131), (self.size, 128, 128, 128),
                               (self.size, 128, 128, 128), (self.size, 64, 259, 259), (self.size, 64, 256, 256),
                               (self.size, 3, 259, 259), (self.size, 64, 256, 256)]

            self.weight_list = [(512*self.size, 512, 3, 3), (3*self.size, 512, 1, 1)] + \
                               [(512*self.size, 512, 3, 3), (512*self.size, 1, 4, 4), (512*self.size, 512, 3, 3),
                                (3, 1, 4, 4), (3*self.size, 512, 1, 1)] * 2 + \
                               [(512, 512, 3, 3), (512, 1, 4, 4), (512, 512, 3, 3), (3, 1, 4, 4), (3, 512, 1, 1),
                                (512, 256, 3, 3), (256, 1, 4, 4), (256, 256, 3, 3), (3, 1, 4, 4), (3, 256, 1, 1),
                                (256, 128, 3, 3), (128, 1, 4, 4), (128, 128, 3, 3), (3, 1, 4, 4), (3, 128, 1, 1),
                                (128, 64, 3, 3), (64, 1, 4, 4), (64, 64, 3, 3), (3, 1, 4, 4), (3, 64, 1, 1)]

            self.all_conv = nn.CellList([self._conv2d(512*self.size, 512*self.size, 3, 1, 1, 1, self.size, 0),
                                         self._conv2d(512*self.size, 3*self.size, 1, 1, 0, 1, self.size, 1),
                                         self._transpose2d(512*self.size, 512*self.size, 3, 2, 0, 1, self.size, 2),
                                         self._conv2d(512*self.size, 512*self.size, 4, 1, 0, 1, 512*self.size, 3),
                                         self._conv2d(512*self.size, 512*self.size, 3, 1, 1, 1, self.size, 4),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 5),
                                         self._conv2d(512*self.size, 3*self.size, 1, 1, 0, 1, self.size, 6),
                                         self._transpose2d(512*self.size, 512*self.size, 3, 2, 0, 1, self.size, 7),
                                         self._conv2d(512*self.size, 512*self.size, 4, 1, 0, 1, 512*self.size, 8),
                                         self._conv2d(512*self.size, 512*self.size, 3, 1, 1, 1, self.size, 9),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 10),
                                         self._conv2d(512*self.size, 3*self.size, 1, 1, 0, 1, self.size, 11),
                                         self._transpose2d(512, 512, 3, 2, 0, 1, 1, 12),
                                         self._conv2d(512, 512, 4, 1, 0, 1, 512, 13),
                                         self._conv2d(512, 512, 3, 1, 1, 1, 1, 14),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 15),
                                         self._conv2d(512, 3, 1, 1, 0, 1, 1, 16),
                                         self._transpose2d(512, 256, 3, 2, 0, 1, 1, 17),
                                         self._conv2d(256, 256, 4, 1, 0, 1, 256, 18),
                                         self._conv2d(256, 256, 3, 1, 1, 1, 1, 19),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 20),
                                         self._conv2d(256, 3, 1, 1, 0, 1, 1, 21),
                                         self._transpose2d(256, 128, 3, 2, 0, 1, 1, 22),
                                         self._conv2d(128, 128, 4, 1, 0, 1, 128, 23),
                                         self._conv2d(128, 128, 3, 1, 1, 1, 1, 24),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 25),
                                         self._conv2d(128, 3, 1, 1, 0, 1, 1, 26),
                                         self._transpose2d(128, 64, 3, 2, 0, 1, 1, 27),
                                         self._conv2d(64, 64, 4, 1, 0, 1, 64, 28),
                                         self._conv2d(64, 64, 3, 1, 1, 1, 1, 29),
                                         self._conv2d(3, 3, 4, 1, 0, 1, 3, 30),
                                         self._conv2d(64, 3, 1, 1, 0, 1, 1, 31)])
        self.all_conv_weight = ms.ParameterTuple(self.all_conv.get_parameters())
        for param in self.all_conv_weight:
            param.requires_grad = False
        self.all_info = [self.all_conv, self.all_conv_weight, self.input_list, self.weight_list, self.name_list]

    def _conv2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, num):
        """
        Conv2d operator.

        Args:
            in_channels (int): Input channel.
            out_channels (int): Output channel.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            dilation(int): Dilation.
            group (int): Group.
            num (int): The number of the operator in all_conv.

        Returns:
            nn.Cell, function of nn.Conv2d with given weight shape.

        Examples:
            >>> func = _conv2d(512, 512, 3, 1, 1, 1, 1, 0)
        """
        if self.dtype_list[num] == 'float32':
            func = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=group,
                             has_bias=False, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float32))
        else:
            func = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=group,
                             has_bias=False, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float16))
        return func

    def _transpose2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, num):
        """
        Transpose2d operator.

        Args:
            in_channels (int): Input channel.
            out_channels (int): Output channel.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            dilation(int): Dilation.
            group (int): Group.
            num (int): The number of the operator in all_conv.

        Returns:
            nn.Cell, function of nn.Conv2dTranspose with given weight shape.

        Examples:
            >>> func = _transpose2d(512, 512, 3, 2, 0, 1, 1, 2)
        """
        if self.dtype_list[num] == 'float32':
            func = nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                                      group=group, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float32))
        else:
            func = nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                                      group=group, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float16))
        return func


    def construct(self, x, img, ws, force_fp32=False, fused_modconv=None, noise_mode=0):
        """Synthesis block construct"""
        unstack = ops.Unstack(axis=1)
        w_iter = iter(unstack(ws))
        d_type = ms.float16 if self.use_fp16 and not force_fp32 else ms.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (d_type == ms.float32 or x.shape[0] == 1)

        d_type = ms.float32
        if self.in_channels != 0:
            x = x.astype(d_type)

        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter).astype(d_type), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           all_info=self.all_info)
        else:
            x = self.conv0(x, next(w_iter).astype(d_type), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           all_info=self.all_info)
            x = self.conv1(x, next(w_iter).astype(d_type), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           all_info=self.all_info)

        if img is not None:
            img = upfirdn2d.upsample2d(img, resample_filter, all_info=self.all_info)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv, all_info=self.all_info)
            y = y.astype(ms.float32)
            img = img + y if img is not None else y

        return x, img


class SynthesisBlockNoPose(nn.Cell):
    """
    Synthesis Block No Pose.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        resolution (int): Resolution of this layer.
        img_channels (int): Number of output color channels.
        is_last (bool): Is this the last block?
        architecture (str): Architecture: 'orig', 'skip'. Default: 'skip'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        use_fp16 (bool): Use FP16 for this block? Default: False.
        batch_size (int): Batch size. Default: 1.
        train (bool): True: train, False: infer. Default: False.
        layer_kwargs (dict): Arguments for SynthesisLayer.

    Inputs:
        - **x** (Tensor) - Input feature.
        - **img** (Tensor) - Input image.
        - **ws** (Tensor) - Intermediate latents.
        - **force_fp32** (bool) - If force the input to float32. Default: False.
        - **fused_modconv** (bool) - Perform modulation, convolution, and demodulation as a single fused operation?
           . Default: True.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.

    Outputs:
        Tensor, output feature.
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> block = SynthesisBlockNoPose(in_channels, out_channels, w_dim, resolution, img_channels, is_last)
        >>> x, img = block(x, img, ws)
    """
    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, is_last, architecture='skip',
                 conv_clamp=None, use_fp16=False, batch_size=1, train=False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.size = batch_size
        self.train = train
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = Parameter(Tensor(np.random.randn(out_channels, resolution, resolution), ms.float32))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        conv_clamp=conv_clamp, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp)
            self.num_torgb += 1

        self.name_list = ['conv2d'] * 2 + (['transpose2d'] + ['conv2d'] * 4) * 6
        self.dtype_list = ['float32' for _ in range(len(self.name_list))]

        self.input_list = [(1, 512*self.size, 4, 4), (1, 512*self.size, 4, 4), (1, 512*self.size, 4, 4),
                           (1, 512*self.size, 11, 11), (1, 512*self.size, 8, 8), (self.size, 3, 11, 11),
                           (1, 512*self.size, 8, 8), (1, 512*self.size, 8, 8), (1, 512*self.size, 19, 19),
                           (1, 512*self.size, 16, 16), (self.size, 3, 19, 19), (1, 512*self.size, 16, 16),
                           (self.size, 512, 16, 16), (self.size, 512, 35, 35), (self.size, 512, 32, 32),
                           (self.size, 3, 35, 35), (self.size, 512, 32, 32), (self.size, 512, 32, 32),
                           (self.size, 256, 67, 67), (self.size, 256, 64, 64), (self.size, 3, 67, 67),
                           (self.size, 256, 64, 64), (self.size, 256, 64, 64), (self.size, 128, 131, 131),
                           (self.size, 128, 128, 128), (self.size, 3, 131, 131), (self.size, 128, 128, 128),
                           (self.size, 128, 128, 128), (self.size, 64, 259, 259), (self.size, 64, 256, 256),
                           (self.size, 3, 259, 259), (self.size, 64, 256, 256)]

        self.weight_list = [(512*self.size, 512, 3, 3), (3*self.size, 512, 1, 1)] + \
                           [(512*self.size, 512, 3, 3), (512*self.size, 1, 4, 4), (512*self.size, 512, 3, 3),
                            (3, 1, 4, 4), (3*self.size, 512, 1, 1)] * 2 + \
                           [(512, 512, 3, 3), (512, 1, 4, 4), (512, 512, 3, 3), (3, 1, 4, 4), (3, 512, 1, 1),
                            (512, 256, 3, 3), (256, 1, 4, 4), (256, 256, 3, 3), (3, 1, 4, 4), (3, 256, 1, 1),
                            (256, 128, 3, 3), (128, 1, 4, 4), (128, 128, 3, 3), (3, 1, 4, 4), (3, 128, 1, 1),
                            (128, 64, 3, 3), (64, 1, 4, 4), (64, 64, 3, 3), (3, 1, 4, 4), (3, 64, 1, 1)]

        self.all_conv = nn.CellList([self._conv2d(512*self.size, 512*self.size, 3, 1, 1, 1, self.size, 0),
                                     self._conv2d(512*self.size, 3*self.size, 1, 1, 0, 1, self.size, 1),
                                     self._transpose2d(512*self.size, 512*self.size, 3, 2, 0, 1, self.size, 2),
                                     self._conv2d(512*self.size, 512*self.size, 4, 1, 0, 1, 512*self.size, 3),
                                     self._conv2d(512*self.size, 512*self.size, 3, 1, 1, 1, self.size, 4),
                                     self._conv2d(3, 3, 4, 1, 0, 1, 3, 5),
                                     self._conv2d(512*self.size, 3*self.size, 1, 1, 0, 1, self.size, 6),
                                     self._transpose2d(512*self.size, 512*self.size, 3, 2, 0, 1, self.size, 7),
                                     self._conv2d(512*self.size, 512*self.size, 4, 1, 0, 1, 512*self.size, 8),
                                     self._conv2d(512*self.size, 512*self.size, 3, 1, 1, 1, self.size, 9),
                                     self._conv2d(3, 3, 4, 1, 0, 1, 3, 10),
                                     self._conv2d(512*self.size, 3*self.size, 1, 1, 0, 1, self.size, 11),
                                     self._transpose2d(512, 512, 3, 2, 0, 1, 1, 12),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 13),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 14), self._conv2d(3, 3, 4, 1, 0, 1, 3, 15),
                                     self._conv2d(512, 3, 1, 1, 0, 1, 1, 16),
                                     self._transpose2d(512, 256, 3, 2, 0, 1, 1, 17),
                                     self._conv2d(256, 256, 4, 1, 0, 1, 256, 18),
                                     self._conv2d(256, 256, 3, 1, 1, 1, 1, 19),
                                     self._conv2d(3, 3, 4, 1, 0, 1, 3, 20),
                                     self._conv2d(256, 3, 1, 1, 0, 1, 1, 21),
                                     self._transpose2d(256, 128, 3, 2, 0, 1, 1, 22),
                                     self._conv2d(128, 128, 4, 1, 0, 1, 128, 23),
                                     self._conv2d(128, 128, 3, 1, 1, 1, 1, 24),
                                     self._conv2d(3, 3, 4, 1, 0, 1, 3, 25),
                                     self._conv2d(128, 3, 1, 1, 0, 1, 1, 26),
                                     self._transpose2d(128, 64, 3, 2, 0, 1, 1, 27),
                                     self._conv2d(64, 64, 4, 1, 0, 1, 64, 28),
                                     self._conv2d(64, 64, 3, 1, 1, 1, 1, 29), self._conv2d(3, 3, 4, 1, 0, 1, 3, 30),
                                     self._conv2d(64, 3, 1, 1, 0, 1, 1, 31)])
        self.all_conv_weight = ms.ParameterTuple(self.all_conv.get_parameters())
        for param in self.all_conv_weight:
            param.requires_grad = False
        self.all_info = [self.all_conv, self.all_conv_weight, self.input_list, self.weight_list, self.name_list]

    def _conv2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, num):
        """
        Conv2d operator.

        Args:
            in_channels (int): Input channel.
            out_channels (int): Output channel.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            dilation(int): Dilation.
            group (int): Group.
            num (int): The number of the operator in all_conv.

        Returns:
            nn.Cell, function of nn.Conv2d with given weight shape.

        Examples:
            >>> func = _conv2d(512, 512, 3, 1, 1, 1, 1, 0)
        """
        if self.dtype_list[num] == 'float32':
            func = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=group,
                             has_bias=False, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float32))
        else:
            func = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=group,
                             has_bias=False, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float16))
        return func

    def _transpose2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, num):
        """
        Transpose2d operator.

        Args:
            in_channels (int): Input channel.
            out_channels (int): Output channel.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            dilation(int): Dilation.
            group (int): Group.
            num (int): The number of the operator in all_conv.

        Returns:
            nn.Cell, function of nn.Conv2dTranspose with given weight shape.

        Examples:
            >>> func = _transpose2d(512, 512, 3, 2, 0, 1, 1, 2)
        """
        if self.dtype_list[num] == 'float32':
            func = nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                                      group=group, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float32))
        else:
            func = nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                                      group=group, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float16))
        return func

    def construct(self, x, img, ws, force_fp32=False, fused_modconv=None, noise_mode=0):
        """Synthesis block no pose construct"""
        unstack = ops.Unstack(axis=1)
        tile = ops.Tile()
        w_iter = iter(unstack(ws))
        d_type = ms.float16 if self.use_fp16 and not force_fp32 else ms.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (d_type == ms.float32 or x.shape[0] == 1)

        d_type = ms.float32
        if self.in_channels == 0:
            x = self.const.astype(d_type)
            x = tile(x.expand_dims(0), (ws.shape[0], 1, 1, 1))
        else:
            x = x.astype(d_type)

        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           all_info=self.all_info)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           all_info=self.all_info)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           all_info=self.all_info)

        if img is not None:
            img = upfirdn2d.upsample2d(img, resample_filter, all_info=self.all_info)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv, all_info=self.all_info)
            y = y.astype(ms.float32)
            img = img + y if img is not None else y

        return x, img


class SynthesisNetwork(nn.Cell):
    """
    Synthesis Network.

    Args:
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): Output image resolution.
        img_channels (int): Number of color channels.
        batch_size (int): Batch size. Default: 1.
        channel_base (int): Overall multiplier for the number of channels. Default: 32768.
        channel_max (int): Maximum number of channels in any layer. Default: 512.
        num_fp16_res (int): Use FP16 for the N highest resolutions. Default: 0.
        train (bool): True: train, False: infer. Default: False.
        **block_kwargs (dict): Arguments for SynthesisBlock.

    Inputs:
        - **ws** (Tensor) - Intermediate latents.
        - **pose** (Tensor) - Pose tensor.
        - **ret_pose** (bool) - Need to return the pose feature. Default: False.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.

    Outputs:
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels, **synthesis_kwargs)
        >>> x = synthesis(ws, pose, ret_pose, **synthesis_kwargs)
    """
    def __init__(self, w_dim, img_resolution, img_channels, batch_size=1, channel_base=32768, channel_max=512,
                 num_fp16_res=0, train=False, **block_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.train = train
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.pose_encoder = PoseEncoder(image_size=self.img_resolution, channel_base=channel_base)
        self.num_ws = 0
        self.blocks = nn.CellList()
        self.num_convs = []
        self.num_torgbs = []
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, img_channels=img_channels,
                                   is_last=is_last, use_fp16=use_fp16, batch_size=batch_size, train=train,
                                   **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            self.blocks.append(block)
            self.num_convs.append(block.num_conv)
            self.num_torgbs.append(block.num_torgb)

    def construct(self, ws, pose, ret_pose=False, noise_mode=0):
        """Synthesis network construct"""
        block_ws = []
        ws = ws.astype(ms.float32)
        w_idx = 0
        for (block, num_conv, num_torgb) in zip(self.blocks, self.num_convs, self.num_torgbs):
            block_ws.append(ws[:, w_idx: w_idx + num_conv + num_torgb, :])
            w_idx += num_conv

        img = None
        pose_enc = self.pose_encoder(pose)
        x = pose_enc[-1]
        for block, cur_ws in zip(self.blocks, block_ws):
            x, img = block(x, img, cur_ws, noise_mode)
        if ret_pose:
            return img, pose_enc
        return img


class SynthesisNetworkNoPose(nn.Cell):
    """
    Synthesis Network No Pose.

    Args:
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): Output image resolution.
        img_channels (int): Number of color channels.
        batch_size (int): Batch size. Default: 1.
        channel_base (int): Overall multiplier for the number of channels. Default: 32768.
        channel_max (int): Maximum number of channels in any layer. Default: 512.
        num_fp16_res (int): Use FP16 for the N highest resolutions. Default: 0.
        train (bool): True: train, False: infer. Default: False.
        **block_kwargs (dict): Arguments for SynthesisBlock.

    Inputs:
        - **ws** (Tensor) - Intermediate latents.
        - **ret_pose** (bool) - Need to return the pose feature. Default: False.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.

    Outputs:
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> synthesis = SynthesisNetworkNoPose(w_dim, img_resolution, img_channels, **synthesis_kwargs)
        >>> x = synthesis(ws, ret_pose, **synthesis_kwargs)
    """
    def __init__(self, w_dim, img_resolution, img_channels, batch_size=1, channel_base=32768, channel_max=512,
                 num_fp16_res=0, train=False, **block_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.train = train
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        self.blocks = nn.CellList()
        self.num_convs = []
        self.num_torgbs = []
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlockNoPose(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                         img_channels=img_channels, is_last=is_last, use_fp16=use_fp16,
                                         batch_size=batch_size, train=train, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            self.blocks.append(block)
            self.num_convs.append(block.num_conv)
            self.num_torgbs.append(block.num_torgb)

    def construct(self, ws, ret_pose=False, noise_mode=0):
        """Synthesis network no pose construct"""
        block_ws = []
        ws = ws.astype(ms.float32)
        w_idx = 0
        for (block, num_conv, num_torgb) in zip(self.blocks, self.num_convs, self.num_torgbs):
            block_ws.append(ws[:, w_idx: w_idx + num_conv + num_torgb, :])
            w_idx += num_conv

        x = img = None
        for block, cur_ws in zip(self.blocks, block_ws):
            x, img = block(x, img, cur_ws, noise_mode)
        if ret_pose:
            return img, None
        return img


class Generator(nn.Cell):
    """
    Generator.

    Args:
        z_dim (int): Input latent (Z) dimensionality.
        c_dim (int): Conditioning label (C) dimensionality.
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): Output image resolution.
        img_channels (int): Number of output color channels.
        batch_size (int): Batch size. Default: 1.
        train (bool): True: train, False: infer. Default: False.
        mapping_kwargs (dict): Arguments for MappingNetwork. Default: None.
        synthesis_kwargs (dict): Arguments for SynthesisNetwork. Default: None.

    Inputs:
        - **z** (Tensor) - Latent tensor.
        - **c** (Tensor) - Label tensor.
        - **pose** (Tensor) - Pose tensor.
        - **truncation_psi** (int) - Truncation coefficient. Default: 1.
        - **truncation_cutoff** (int) - Truncation cutoff if truncation_psi != 1. Default: None.
        - **ret_pose** (bool) - Need to return the pose feature. Default: False.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.

    Outputs:
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> generator = Generator(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs)
        >>> x = generator(z, c, pose)
    """
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels, batch_size=1, train=False,
                 mapping_kwargs=None, synthesis_kwargs=None):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.train = train
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                          batch_size=batch_size, train=train, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def construct(self, z, c, pose, truncation_psi=1, truncation_cutoff=None, ret_pose=False, noise_mode=0):
        """Generator construct"""
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        return self.synthesis(ws, pose, ret_pose=ret_pose, noise_mode=noise_mode)


class GeneratorNoPose(nn.Cell):
    """
    GeneratorNoPose.

    Args:
        z_dim (int): Input latent (Z) dimensionality.
        c_dim (int): Conditioning label (C) dimensionality.
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): Output image resolution.
        img_channels (int): Number of output color channels.
        batch_size (int): Batch size. Default: 1.
        train (bool): True: train, False: infer. Default: False.
        mapping_kwargs (dict): Arguments for MappingNetwork. Default: None.
        synthesis_kwargs (dict): Arguments for SynthesisNetwork. Default: None.

    Inputs:
        - **z** (Tensor) - Latent tensor.
        - **c** (Tensor) - Label tensor.
        - **truncation_psi** (int) - Truncation coefficient. Default: 1.
        - **truncation_cutoff** (int) - Truncation cutoff if truncation_psi != 1. Default: None.
        - **ret_pose** (bool) - Need to return the pose feature. Default: False.
        - **noise_mode** (int) - Noise mode 0: const, 1: random. Default: 0.

    Outputs:
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> generator = GeneratorNoPose(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs)
        >>> x = generator(z, c)
    """
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels, batch_size=1, train=False,
                 mapping_kwargs=None, synthesis_kwargs=None):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.train = train
        self.synthesis = SynthesisNetworkNoPose(w_dim=w_dim, img_resolution=img_resolution,
                                                img_channels=img_channels, batch_size=batch_size,
                                                train=train, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def construct(self, z, c, truncation_psi=1, truncation_cutoff=None, ret_pose=False, noise_mode=0):
        """GeneratorNoPose construct"""
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        return self.synthesis(ws, ret_pose=ret_pose, noise_mode=noise_mode)
