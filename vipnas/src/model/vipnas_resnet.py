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
"""Define ViPNASResNet backbone"""

import copy

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal, HeNormal, HeUniform, Constant


class ViPNASBottleneck(nn.Cell):
    """Bottleneck block for ViPNASResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Cell): downsample operation on identity branch.
            Default: None.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: None.
        kernel_size (int): kernel size of conv2 searched in ViPANS.
        groups (int): group number of conv2 searched in ViPNAS.
        attention (bool): whether to use attention module in the end of
            the block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 kernel_size=3,
                 groups=1,
                 attention=False):
        # Protect mutable default arguments
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_cfg = copy.deepcopy(norm_cfg)
        super(ViPNASBottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.conv1_stride = 1
        self.conv2_stride = stride

        if norm_cfg['type'] == 'BN':
            self.norm1 = nn.BatchNorm2d(num_features=self.mid_channels)
            self.norm2 = nn.BatchNorm2d(num_features=self.mid_channels)
            self.norm3 = nn.BatchNorm2d(num_features=out_channels)
        else:
            raise ValueError('norm_cfg type not support yet')

        if conv_cfg is None or conv_cfg['type'] == 'Conv2d' or conv_cfg['type'] == 'Conv':
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.mid_channels,
                kernel_size=1,
                stride=self.conv1_stride,
                pad_mode='valid',
                weight_init=Normal(sigma=0.001))
            self.conv2 = nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=kernel_size,
                stride=self.conv2_stride,
                pad_mode='pad',
                weight_init=Normal(sigma=0.001),
                padding=kernel_size // 2,
                group=groups,
                dilation=dilation)
            self.conv3 = nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                pad_mode='valid',
                weight_init=Normal(sigma=0.001))
        else:
            raise ValueError('Conv_cfg type not support yet')

        if attention:
            self.attention = ContextBlock(out_channels, max(1.0 / 16, 16.0 / out_channels))
        else:
            self.attention = None

        self.relu = nn.ReLU()
        self.downsample = downsample

    def construct(self, x):
        """Construct function."""

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.attention is not None:
            out = self.attention.construct(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       4 for ``ViPNAS_Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, ViPNASBottleneck):
            expansion = 1
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ViPNASResLayer(nn.SequentialCell):
    """ViPNASResLayer to build ResNet style backbone.

    Args:
        block (type): Residual block used to build ViPNAS ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        kernel_size (int): Kernel Size of the corresponding convolution layer
            searched in the block.
        groups (int): Group number of the corresponding convolution layer
            searched in the block.
        attention (bool): Whether to use attention module in the end of the
            block.
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 kernel_size=3,
                 groups=1,
                 attention=False,
                 **kwargs):
        # Protect mutable default arguments
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_cfg = copy.deepcopy(norm_cfg)
        self.expansion = get_expansion(block, expansion)

        downsample = []
        if stride != 1 or in_channels != out_channels:
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        pad_mode='valid'))
            if norm_cfg['type'] == 'BN':
                downsample.extend([
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=conv_stride,
                        pad_mode='valid',
                        weight_init=Normal(sigma=0.001)),
                    nn.BatchNorm2d(num_features=out_channels)
                ])
            else:
                raise ValueError('norm_cfg type not support yet')
            downsample = nn.SequentialCell(*downsample)

        layers = [block(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion=self.expansion,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            kernel_size=kernel_size,
            groups=groups,
            attention=attention,
            **kwargs)]
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    kernel_size=kernel_size,
                    groups=groups,
                    attention=attention,
                    **kwargs))

        super(ViPNASResLayer, self).__init__(*layers)
        self.layers = layers

    def construct(self, x):
        if self.layers:
            for layer in self.layers:
                x = layer(x)
        out = x

        return out


class ViPNASResNet(nn.Cell):
    """ViPNASResNet backbone.

    ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search.
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        wid (list(int)): Searched width config for each stage.Default: None.
        expan (list(int)): Searched expansion ratio config for each stage.Default: None.
        dep (list(int)): Searched depth config for each stage.Default: None.
        ks (list(int)): Searched kernel size config for each stage.Default: None.
        group (list(int)): Searched group number config for each stage.Default: None.
        att (list(bool)): Searched attention config for each stage. Default: None.
    """

    arch_settings = {
        50: ViPNASBottleneck,
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_eval=False,
                 zero_init_residual=True,
                 wid=None,
                 expan=None,
                 dep=None,
                 ks=None,
                 group=None,
                 att=None):
        # Protect mutable default arguments
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if wid is None:
            wid = [48, 80, 160, 304, 608]
        if expan is None:
            expan = [0, 1, 1, 1, 1]
        if dep is None:
            dep = [0, 4, 6, 7, 3]
        if ks is None:
            ks = [7, 3, 5, 5, 5]
        if group is None:
            group = [0, 16, 16, 16, 16]
        if att is None:
            att = [False, True, False, True, True]
        norm_cfg = copy.deepcopy(norm_cfg)
        super(ViPNASResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = dep[0]
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.avg_down = avg_down
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block = self.arch_settings[depth]
        self.stage_blocks = dep[1:1 + num_stages]

        self._make_stem_layer(in_channels, wid[0], ks[0])

        self.res_layers = nn.CellList()
        layer_list = []
        in_channels = wid[0]
        res_layer = None
        for i, num_blocks in enumerate(self.stage_blocks):
            expansion = get_expansion(self.block, expan[i + 1])
            out_channels = wid[i + 1] * expansion
            stride = strides[i]
            dilation = dilations[i]
            res_layer = ViPNASResLayer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=expansion,
                stride=stride,
                dilation=dilation,
                avg_down=self.avg_down,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                kernel_size=ks[i + 1],
                groups=group[i + 1],
                attention=att[i + 1])
            in_channels = out_channels
            layer_list.append(res_layer)

        self.seq_layers = nn.CellList(layer_list)
        self.feat_dim = res_layer[-1].out_channels

    def _make_stem_layer(self, in_channels, stem_channels, kernel_size):
        """Make stem layer."""

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=kernel_size,
            stride=2,
            pad_mode='pad',
            weight_init=Normal(sigma=0.001),
            padding=kernel_size // 2)
        self.norm1 = nn.BatchNorm2d(num_features=stem_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

    def construct(self, x):
        """Construct function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        pad_op = ops.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        x = pad_op(x)
        x = self.maxpool(x)
        outs = []
        i = 0
        for block in self.seq_layers:
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
            i += 1
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        module.weight = initializer(val, module.weight.shape, mindspore.float32)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias = initializer(bias, module.bias.shape, mindspore.float32)
    if hasattr(module, 'gamma') and module.gamma is not None:
        module.gamma = initializer(val, module.gamma.shape, mindspore.float32)
    if hasattr(module, 'beta') and module.beta is not None:
        module.beta = initializer(bias, module.beta.shape, mindspore.float32)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    """Define Kaiming_init"""
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            he_uniform = HeUniform(
                negative_slope=a, mode=mode, nonlinearity=nonlinearity)
            module.weight = initializer(he_uniform, module.weight.shape, mindspore.float32)
        else:
            he_normal = HeNormal(
                negative_slope=a, mode=mode, nonlinearity=nonlinearity)
            module.weight = initializer(he_normal, module.weight.shape, mindspore.float32)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias = initializer(Constant(bias), module.bias.shape, mindspore.float32)


def last_zero_init(m):
    if isinstance(m, nn.SequentialCell):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Cell):
    """ContextBlock module in GCNet.
    See 'GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond'
    (https://arxiv.org/abs/1904.11492) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling.
            Options are 'att' and 'avg', stand for attention pooling and
            average pooling respectively. Default: 'att'.
        fusion_types (Sequence[str]): Fusion method for feature fusion,
            Options are 'channels_add', 'channel_mul', stand for channelwise
            addition and multiplication respectively. Default: ('channel_add',)
    """

    _abbr_ = 'context_block'

    def __init__(self,
                 in_channels,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(
                in_channels,
                1,
                kernel_size=1,
                pad_mode='valid',
                weight_init=Normal(sigma=0.001),
                has_bias=True,
                bias_init='zeros')
            self.softmax = nn.Softmax(axis=2)
        else:
            self.avg_pool = ops.AdaptiveAvgPool2D(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.SequentialCell(
                nn.Conv2d(
                    self.in_channels,
                    self.planes,
                    kernel_size=1,
                    pad_mode='valid',
                    weight_init=Normal(sigma=0.001),
                    has_bias=True,
                    bias_init='zeros'),
                nn.LayerNorm([self.planes, 1, 1], begin_norm_axis=1, begin_params_axis=1),
                nn.ReLU(),  # yapf: disable
                nn.Conv2d(
                    self.planes,
                    self.in_channels,
                    kernel_size=1,
                    pad_mode='valid',
                    weight_init=Normal(sigma=0.001),
                    has_bias=True,
                    bias_init='zeros'))
        else:
            self.channel_add_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        """Define spatial_pool"""
        batch, channel, height, width = x.shape
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.expand_dims(axis=1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.expand_dims(axis=3)
            # [N, 1, C, 1]
            cast = ops.Cast()
            input_x = cast(input_x, mindspore.float16)
            context = ops.matmul(input_x, context_mask)
            context = cast(context, mindspore.float32)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def construct(self, x):
        """Construct ContextBlock module"""
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
