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
"""
StyleGAN2 Generator
"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Parameter

from model.block import normalize_2nd_moment, FullyConnectedLayer, SynthesisBlock


class MappingNetwork(nn.Cell):
    """
    Mapping Network.

    Args:
        z_dim (int): Input latent (Z) dimensionality, 0 = no latent.
        c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
        w_dim (int): Intermediate latent (W) dimensionality.
        num_ws (int): Number of intermediate latents to output, None = do not broadcast.
        num_layers (int): Number of mapping layers. Default: 8.
        embed_features (bool): Label embedding dimensionality Default: None.
        layer_features (bool): Number of intermediate features in the mapping layers.
            None = same as w_dim. Default: None.
        activation (str): Activation function: 'relu', 'lrelu', etc. Default: 'lrelu'.
        lr_multiplier (float): Learning rate multiplier for the mapping layers. Default: 0.01.
        w_avg_beta (float): Decay for tracking the moving average of W during training,
            None = do not track. Default: 0.995.

    Input:
        - **z** (Tensor) - Latent tensor.
        - **c** (Tensor) - Label tensor.
        - **truncation_psi** (int) - GAN truncation trick. Default: 0.5
        - **truncation_cutoff (int) - Cutoff for truncation. Default: None
        - **skip_w_avg_update** (bool) - Skip update for moving average of weight. Default: False

    Output:
        Tensor, mapping network output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        mapping = MappingNetwork(z_dim, c_dim, w_dim, num_ws)
        ws = mapping(z, c)
    """

    def __init__(self, z_dim, c_dim, w_dim, num_ws, num_layers=8, embed_features=None,
                 layer_features=None, activation='lrelu', lr_multiplier=0.01, w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.layer = nn.CellList()

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            self.layer.append(layer)

        if num_ws is not None and w_avg_beta is not None:
            self.w_avg = Parameter(ops.Zeros()(w_dim, ms.float32))

    def construct(self, z, c, truncation_psi=0.5, truncation_cutoff=None, skip_w_avg_update=False):
        """Mapping network construct"""
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z.astype(ms.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.astype(ms.float32)))
            x = ops.Concat(1)((x, y)) if x is not None else y

        # Main layers.
        for layer in self.layer:
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg = (x.mean(axis=0) + (self.w_avg - x.mean(axis=0)) * self.w_avg_beta).copy()

        # Broadcast.
        if self.num_ws is not None:
            x = x.expand_dims(1).repeat(self.num_ws, 1)

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg + (x - self.w_avg) * truncation_psi
            else:
                x[:, :truncation_cutoff] = self.w_avg + (x[:, :truncation_cutoff] - self.w_avg) * truncation_psi
        return x


class SynthesisNetwork(nn.Cell):
    """
    Synthesis Network.

    Args:
        w_dim (int): Intermediate latent (W) dimensionality.
        img_resolution (int): Output image resolution.
        img_channels (int): Number of color channels.
        channel_base (int): Overall multiplier for the number of channels. Default: 32768.
        channel_max (int): Maximum number of channels in any layer. Default: 512.
        num_fp16_res (int): Use FP16 for the N highest resolutions. Default: 0.
        batch_size (int): Batch size. Default: 1.
        train (bool): True = train, False = infer. Default: False.
        block_kwargs (dict): Arguments for SynthesisBlock.

    Input:
        - **ws** (Tensor) - Intermediate latents.
        - **block_kwargs** (dict) - Arguments for SynthesisBlock.

    Output:
       Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels, **block_kwargs)
        >>> x = synthesis(ws, **block_kwargs)
    """

    def __init__(self, w_dim, img_resolution, img_channels, channel_base=32768, channel_max=512,
                 num_fp16_res=0, batch_size=1, train=False, **block_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.train = train
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.num_ws = 0
        self.num_convs = []
        self.num_torgbs = []
        self.block = nn.CellList()

        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, output_res=img_resolution,
                                   img_channels=img_channels, is_last=is_last, use_fp16=use_fp16,
                                   batch_size=batch_size, train=train, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            self.block.append(block)
            self.num_convs.append(block.num_conv)
            self.num_torgbs.append(block.num_torgb)

    def construct(self, ws, **block_kwargs):
        """Synthesis network construct"""
        block_ws = []
        ws = ws.astype(ms.float32)
        w_idx = 0
        for (block, num_conv, num_torgb) in zip(self.block, self.num_convs, self.num_torgbs):
            block_ws.append(ws[:, w_idx: w_idx + num_conv + num_torgb, :])
            w_idx += num_conv

        x = img = None
        for block, cur_ws in zip(self.block, block_ws):
            x, img = block(x, img, cur_ws, **block_kwargs)
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

    Input:
        - **z** (Tensor) - Latent tensor.
        - **c** (Tensor) - Label tensor.
        - **truncation_psi** (int) - GAN truncation trick. Default: 0.5
        - **truncation_cutoff** (int) - Cutoff for truncation. Default: None
        - **synthesis_kwargs** (dict) - Arguments for SynthesisNetwork.

    Output:
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> generator = Generator(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs, synthesis_kwargs)
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
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                          batch_size=batch_size, train=train, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def construct(self, z, c, truncation_psi=0.5, truncation_cutoff=None, **synthesis_kwargs):
        """Generator construct"""
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img
