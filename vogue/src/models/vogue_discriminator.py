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
"""vogue_discriminator_block"""

import numpy as np
from mindspore import nn, ops, Tensor
import mindspore as ms

from vogue_utils import upfirdn2d
from models.vogue_block import Conv2dLayer, FullyConnectedLayer, MappingNetwork, resample_filter


class DiscriminatorBlock(nn.Cell):
    """
    Discriminator Block.

    Args:
        in_channels (int): Number of input channels, 0 = first block.
        tmp_channels (int): Number of intermediate channels.
        out_channels (int): Number of output channels.
        resolution (int): Resolution of this block.
        img_channels (int): Number of input color channels.
        first_layer_idx (int): Index of the first layer.
        architecture (str): Architecture: 'orig', 'skip', 'resnet'. Default: 'resnet'.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'lrelu'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        use_fp16 (bool): Use FP16 for this block? Default: False.
        fp16_channels_last (bool): Use channels-last memory format with FP16? Default: False.
        freeze_layers (int): Number of layers to freeze. Default: 0.

    Inputs:
        - **x** (Tensor) - Input feature.
        - **img** (Tensor) - Input image.
        - **all_info** (list) - Information of all_conv. Default: None.

    Outputs:
        Tensor, discriminator block output feature.
        Tensor, discriminator block output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution, first_layer_idx)
        >>> x, img = block(x, img, all_info=all_info)
    """
    def __init__(self, in_channels, tmp_channels, out_channels, resolution, img_channels, first_layer_idx,
                 architecture='resnet', activation='lrelu', conv_clamp=None, use_fp16=False, fp16_channels_last=False,
                 freeze_layers=0):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                                       trainable=next(trainable_iter), conv_clamp=conv_clamp)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
                                 trainable=next(trainable_iter), conv_clamp=conv_clamp)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation,
                                 down=2, trainable=next(trainable_iter), conv_clamp=conv_clamp)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False,
                                    down=2, trainable=next(trainable_iter))

    def construct(self, x, img, all_info=None):
        """discriminator block construct"""
        d_type = ms.float32

        if x is not None:
            x = x.astype(d_type)

        if self.in_channels == 0 or self.architecture == 'skip':
            img = img.astype(d_type)
            y = self.fromrgb(img, all_info=all_info)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, resample_filter, all_info=all_info) \
                if self.architecture == 'skip' else None

        if self.architecture == 'resnet':
            y = self.skip(x, gain=0.707, all_info=all_info)
            x = self.conv0(x, all_info=all_info)
            x = self.conv1(x, gain=0.707, all_info=all_info)
            x = y + x
        else:
            x = self.conv0(x, all_info=all_info)
            x = self.conv1(x, all_info=all_info)
        return x, img


class MinibatchStdLayer(nn.Cell):
    """
    Mini-batch Std Layer.

    Args:
        group_size (int): Group size of mini-batch.
        num_channels (int): Channel number. Default: 1.
        batch_size (int): Batch size. Default: 1.

    Inputs:
        - **x** (Tensor) - Input tensor.

    Outputs:
        Tensor, mini-batch std layer output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = MinibatchStdLayer(group_size, num_channels)
        >>> x = layer(x)
    """
    def __init__(self, group_size, num_channels=1, batch_size=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.g1 = min(self.group_size, self.batch_size) if self.group_size else self.batch_size

    def construct(self, x):
        """mini-batch std layer construct"""
        square = ops.Square()
        sqrt = ops.Sqrt()
        tile = ops.Tile()
        concat = ops.Concat(axis=1)
        _, c1, h1, w1 = x.shape
        g1 = self.g1
        f1 = self.num_channels
        c = c1 // f1

        y = x.reshape(g1, -1, f1, c, h1, w1)
        y = y - y.mean(axis=0)
        y = square(y).mean(axis=0)
        y = sqrt(y + 1e-8)
        y = y.mean(axis=[2, 3, 4])
        y = y.reshape(-1, f1, 1, 1)
        y = tile(y, (g1, 1, h1, w1))
        x = concat((x, y))
        return x


class DiscriminatorEpilogue(nn.Cell):
    """
    Discriminator Epilogue Block.

    Args:
        in_channels (int): Number of input channels.
        cmap_dim (int): Dimensionality of mapped conditioning label, 0 = no label.
        resolution (int): Resolution of this block.
        img_channels (int): Number of input color channels.
        architecture (str): Architecture: 'orig', 'skip', 'resnet'. Default: 'resnet'.
        batch_size (int): Batch size. Default: 1.
        mbstd_group_size (int): Group size for the minibatch standard deviation layer,
            None = entire minibatch. Default: 4.
        mbstd_num_channels (int): Number of features for the minibatch standard deviation
            layer, 0 = disable. Default: 1.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'lrelu'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.

    Inputs:
        - **x** (Tensor) - Input feature.
        - **img** (Tensor) - Input image.
        - **cmap** (Tensor) - Mapping network output tensor.
        - **all_info** (list) - Information of all_conv. Default: None.

    Outputs:
        Tensor, discriminator epilogue output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> block = DiscriminatorEpilogue(in_channels, cmap_dim, resolution)
        >>> x = block(x, img, camp, all_info=all_info)
    """
    def __init__(self, in_channels, cmap_dim, resolution, img_channels, architecture='resnet', batch_size=1,
                 mbstd_group_size=4, mbstd_num_channels=1, activation='lrelu', conv_clamp=None):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.batch_size = batch_size

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels,
                                       batch_size=batch_size) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3,
                                activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def construct(self, x, img, cmap, all_info=None):
        """ discriminator epilogue construct"""
        flatten = ops.Flatten()
        d_type = ms.float32

        x = x.astype(d_type)
        if self.architecture == 'skip':
            img = img.astype(d_type)
            x = x + self.fromrgb(img, all_info=all_info)

        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x, all_info=all_info)
        x = self.fc(flatten(x))
        x = self.out(x)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(axis=1, keepdims=True) * (1 / np.sqrt(self.cmap_dim))
        return x


class Discriminator(nn.Cell):
    """
    Discriminator Block.

    Args:
        c_dim (int): Conditioning label (C) dimensionality.
        img_resolution (int): Input resolution.
        img_channels (int): Number of input color channels.
        architecture (str): Architecture: 'orig', 'skip', 'resnet'. Default: 'resnet'.
        channel_base (int): Overall multiplier for the number of channels. Default: 32768.
        channel_max (int):  Maximum number of channels in any layer. Default: 512.
        num_fp16_res (int): Use FP16 for the N highest resolutions. Default: 0.
        batch_size (int): Batch size. Default: 1.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        cmap_dim (bool): Dimensionality of mapped conditioning label. Default: None.
        block_kwargs (dict): Arguments for DiscriminatorBlock. Default: None.
        mapping_kwargs (dict): Arguments for MappingNetwork. Default: None.
        epilogue_kwargs (dict): Arguments for DiscriminatorEpilogue. Default: None.

    Inputs:
        - **img** (Tensor) - Input image.
        - **pose** (Tensor) - Pose tensor.
        - **c** (Tensor) - Label tensor.

    Outputs:
        Tensor, discriminator output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> discriminator = Discriminator(c_dim, img_resolution, img_channels)
        >>> x = discriminator(img, pose, c)
    """
    def __init__(self, c_dim, img_resolution, img_channels, architecture='resnet', channel_base=32768,
                 channel_max=512, num_fp16_res=0, batch_size=1, conv_clamp=None, cmap_dim=None, block_kwargs=None,
                 mapping_kwargs=None, epilogue_kwargs=None):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.size = batch_size
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        self.blocks = nn.CellList()
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16,
                                       **block_kwargs, **common_kwargs)
            self.blocks.append(block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, batch_size=batch_size,
                                        **epilogue_kwargs, **common_kwargs)

        self.input_list = [(self.size, 3, 256, 256), (self.size, 64, 258, 258), (self.size, 64, 128, 128),
                           (self.size, 64, 256, 256), (self.size, 64, 260, 260), (self.size, 64, 257, 257),
                           (self.size, 128, 130, 130), (self.size, 128, 64, 64), (self.size, 128, 128, 128),
                           (self.size, 128, 132, 132), (self.size, 128, 129, 129), (self.size, 256, 66, 66),
                           (self.size, 256, 32, 32), (self.size, 256, 64, 64), (self.size, 256, 68, 68),
                           (self.size, 256, 65, 65), (self.size, 512, 34, 34), (self.size, 512, 16, 16),
                           (self.size, 512, 32, 32), (self.size, 512, 36, 36), (self.size, 512, 33, 33),
                           (self.size, 512, 18, 18), (self.size, 512, 8, 8), (self.size, 512, 16, 16),
                           (self.size, 512, 20, 20), (self.size, 512, 17, 17), (self.size, 512, 10, 10),
                           (self.size, 512, 4, 4), (self.size, 512, 8, 8), (self.size, 512, 12, 12),
                           (self.size, 512, 9, 9), (self.size, 513, 4, 4)]

        self.weight_list = [(64, 3, 1, 1), (64, 1, 4, 4), (128, 64, 1, 1), (64, 64, 3, 3), (64, 1, 4, 4),
                            (128, 64, 3, 3), (128, 1, 4, 4), (256, 128, 1, 1), (128, 128, 3, 3), (128, 1, 4, 4),
                            (256, 128, 3, 3), (256, 1, 4, 4), (512, 256, 1, 1), (256, 256, 3, 3), (256, 1, 4, 4),
                            (512, 256, 3, 3)] + \
                           [(512, 1, 4, 4), (512, 512, 1, 1), (512, 512, 3, 3), (512, 1, 4, 4), (512, 512, 3, 3)] * 3 \
                           + [(512, 513, 3, 3)]

        self.name_list = ['conv2d' for _ in range(len(self.input_list))]
        self.dtype_list = ['float32' for _ in range(len(self.input_list))]

        self.all_conv = nn.CellList([self._conv2d(3, 64, 1, 1, 0, 1, 1, 0),
                                     self._conv2d(64, 64, 4, 1, 0, 1, 64, 1),
                                     self._conv2d(64, 128, 1, 1, 0, 1, 1, 2),
                                     self._conv2d(64, 64, 3, 1, 1, 1, 1, 3),
                                     self._conv2d(64, 64, 4, 1, 0, 1, 64, 4),
                                     self._conv2d(64, 128, 3, 2, 0, 1, 1, 5),
                                     self._conv2d(128, 128, 4, 1, 0, 1, 128, 6),
                                     self._conv2d(128, 256, 1, 1, 0, 1, 1, 7),
                                     self._conv2d(128, 128, 3, 1, 1, 1, 1, 8),
                                     self._conv2d(128, 128, 4, 1, 0, 1, 128, 9),
                                     self._conv2d(128, 256, 3, 2, 0, 1, 1, 10),
                                     self._conv2d(256, 256, 4, 1, 0, 1, 256, 11),
                                     self._conv2d(256, 512, 1, 1, 0, 1, 1, 12),
                                     self._conv2d(256, 256, 3, 1, 1, 1, 1, 13),
                                     self._conv2d(256, 256, 4, 1, 0, 1, 256, 14),
                                     self._conv2d(256, 512, 3, 2, 0, 1, 1, 15),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 16),
                                     self._conv2d(512, 512, 1, 1, 0, 1, 1, 17),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 18),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 19),
                                     self._conv2d(512, 512, 3, 2, 0, 1, 1, 20),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 21),
                                     self._conv2d(512, 512, 1, 1, 0, 1, 1, 22),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 23),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 24),
                                     self._conv2d(512, 512, 3, 2, 0, 1, 1, 25),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 26),
                                     self._conv2d(512, 512, 1, 1, 0, 1, 1, 27),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 28),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 29),
                                     self._conv2d(512, 512, 3, 2, 0, 1, 1, 30),
                                     self._conv2d(513, 512, 3, 1, 1, 1, 1, 31)])
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
            >>> func = _conv2d(3, 64, 1, 1, 0, 1, 1, 0)
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

    def construct(self, img, c, **block_kwargs):
        """discriminator construct"""
        x = None
        for block in self.blocks:
            x, img = block(x, img, all_info=self.all_info, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c, all_info=self.all_info)
        x = self.b4(x, img, cmap, all_info=self.all_info)
        return x


class DiscriminatorNoPose(nn.Cell):
    """
    DiscriminatorNoPose Block.

    Args:
        c_dim (int): Conditioning label (C) dimensionality.
        img_resolution (int): Input resolution.
        img_channels (int): Number of input color channels.
        architecture (str): Architecture: 'orig', 'skip', 'resnet'. Default: 'resnet'.
        channel_base (int): Overall multiplier for the number of channels. Default: 32768.
        channel_max (int):  Maximum number of channels in any layer. Default: 512.
        num_fp16_res (int): Use FP16 for the N highest resolutions. Default: 0.
        batch_size (int): Batch size. Default: 1.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        cmap_dim (bool): Dimensionality of mapped conditioning label. Default: None.
        block_kwargs (dict): Arguments for DiscriminatorBlock. Default: None.
        mapping_kwargs (dict): Arguments for MappingNetwork. Default: None.
        epilogue_kwargs (dict): Arguments for DiscriminatorEpilogue. Default: None.

    Inputs:
        - **img** (Tensor) - Input image.
        - **c** (Tensor) - Label tensor.

    Outputs:
        Tensor, discriminator output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> discriminator = DiscriminatorNoPose(c_dim, img_resolution, img_channels)
        >>> x = discriminator(img, c)
    """
    def __init__(self, c_dim, img_resolution, img_channels, architecture='resnet', channel_base=32768,
                 channel_max=512, num_fp16_res=0, batch_size=1, conv_clamp=None, cmap_dim=None, block_kwargs=None,
                 mapping_kwargs=None, epilogue_kwargs=None):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.size = batch_size
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        self.blocks = nn.CellList()
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16,
                                       **block_kwargs, **common_kwargs)
            self.blocks.append(block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        **common_kwargs)

        self.input_list = [(self.size, 3, 256, 256), (self.size, 64, 258, 258), (self.size, 64, 128, 128),
                           (self.size, 64, 256, 256), (self.size, 64, 260, 260), (self.size, 64, 257, 257),
                           (self.size, 128, 130, 130), (self.size, 128, 64, 64), (self.size, 128, 128, 128),
                           (self.size, 128, 132, 132), (self.size, 128, 129, 129), (self.size, 256, 66, 66),
                           (self.size, 256, 32, 32), (self.size, 256, 64, 64), (self.size, 256, 68, 68),
                           (self.size, 256, 65, 65), (self.size, 512, 34, 34), (self.size, 512, 16, 16),
                           (self.size, 512, 32, 32), (self.size, 512, 36, 36), (self.size, 512, 33, 33),
                           (self.size, 512, 18, 18), (self.size, 512, 8, 8), (self.size, 512, 16, 16),
                           (self.size, 512, 20, 20), (self.size, 512, 17, 17), (self.size, 512, 10, 10),
                           (self.size, 512, 4, 4), (self.size, 512, 8, 8), (self.size, 512, 12, 12),
                           (self.size, 512, 9, 9), (self.size, 513, 4, 4)]

        self.weight_list = [(64, 3, 1, 1), (64, 1, 4, 4), (128, 64, 1, 1), (64, 64, 3, 3), (64, 1, 4, 4),
                            (128, 64, 3, 3), (128, 1, 4, 4), (256, 128, 1, 1), (128, 128, 3, 3), (128, 1, 4, 4),
                            (256, 128, 3, 3), (256, 1, 4, 4), (512, 256, 1, 1), (256, 256, 3, 3), (256, 1, 4, 4),
                            (512, 256, 3, 3)] + \
                           [(512, 1, 4, 4), (512, 512, 1, 1), (512, 512, 3, 3), (512, 1, 4, 4), (512, 512, 3, 3)] * 3 \
                           + [(512, 513, 3, 3)]

        self.name_list = ['conv2d' for _ in range(len(self.input_list))]
        self.dtype_list = ['float32' for _ in range(len(self.input_list))]

        self.all_conv = nn.CellList([self._conv2d(3, 64, 1, 1, 0, 1, 1, 0),
                                     self._conv2d(64, 64, 4, 1, 0, 1, 64, 1),
                                     self._conv2d(64, 128, 1, 1, 0, 1, 1, 2),
                                     self._conv2d(64, 64, 3, 1, 1, 1, 1, 3),
                                     self._conv2d(64, 64, 4, 1, 0, 1, 64, 4),
                                     self._conv2d(64, 128, 3, 2, 0, 1, 1, 5),
                                     self._conv2d(128, 128, 4, 1, 0, 1, 128, 6),
                                     self._conv2d(128, 256, 1, 1, 0, 1, 1, 7),
                                     self._conv2d(128, 128, 3, 1, 1, 1, 1, 8),
                                     self._conv2d(128, 128, 4, 1, 0, 1, 128, 9),
                                     self._conv2d(128, 256, 3, 2, 0, 1, 1, 10),
                                     self._conv2d(256, 256, 4, 1, 0, 1, 256, 11),
                                     self._conv2d(256, 512, 1, 1, 0, 1, 1, 12),
                                     self._conv2d(256, 256, 3, 1, 1, 1, 1, 13),
                                     self._conv2d(256, 256, 4, 1, 0, 1, 256, 14),
                                     self._conv2d(256, 512, 3, 2, 0, 1, 1, 15),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 16),
                                     self._conv2d(512, 512, 1, 1, 0, 1, 1, 17),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 18),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 19),
                                     self._conv2d(512, 512, 3, 2, 0, 1, 1, 20),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 21),
                                     self._conv2d(512, 512, 1, 1, 0, 1, 1, 22),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 23),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 24),
                                     self._conv2d(512, 512, 3, 2, 0, 1, 1, 25),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 26),
                                     self._conv2d(512, 512, 1, 1, 0, 1, 1, 27),
                                     self._conv2d(512, 512, 3, 1, 1, 1, 1, 28),
                                     self._conv2d(512, 512, 4, 1, 0, 1, 512, 29),
                                     self._conv2d(512, 512, 3, 2, 0, 1, 1, 30),
                                     self._conv2d(513, 512, 3, 1, 1, 1, 1, 31)])
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
            >>> func = _conv2d(3, 64, 1, 1, 0, 1, 1, 0)
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

    def construct(self, img, c, **block_kwargs):
        """DiscriminatorNoPose construct"""
        x = None
        for block in self.blocks:
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x
