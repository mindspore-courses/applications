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
"""Define ViPNASMobileNetV3 backbone"""

import copy
import warnings
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal, HeNormal, HeUniform, Constant


class ViPNASMobileNetV3(nn.Cell):
    """ViPNASMobileNetV3 backbone.

    ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search.
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    Args:
        wid (list(int)): Searched width config for each stage.Default: None.
        expan (list(int)): Searched expansion ratio config for each stage.Default: None.
        dep (list(int)): Searched depth config for each stage.Default: None.
        ks (list(int)): Searched kernel size config for each stage.Default: None.
        group (list(int)): Searched group number config for each stage.Default: None.
        att (list(bool)): Searched attention config for each stage.Default: None.
        stride (list(int)): Stride config for each stage.Default: None.
        act (list(dict)): Activation config for each stage.Default: None.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
    """

    def __init__(self,
                 wid=None,
                 expan=None,
                 dep=None,
                 ks=None,
                 group=None,
                 att=None,
                 stride=None,
                 act=None,
                 conv_cfg=None,
                 norm_cfg=None):
        # Protect mutable default arguments
        if expan is None:
            expan = [0, 1, 5, 4, 5, 5, 6]
        if dep is None:
            dep = [0, 1, 4, 4, 4, 4, 4]
        if ks is None:
            ks = [3, 3, 7, 7, 5, 7, 5]
        if group is None:
            group = [0, 8, 120, 20, 100, 280, 240]
        if att is None:
            att = [False, True, True, False, True, True, True]
        if stride is None:
            stride = [2, 1, 2, 2, 2, 1, 2]
        if act is None:
            act = [
                'HSwish', 'ReLU', 'ReLU', 'ReLU', 'HSwish', 'HSwish', 'HSwish'
            ]
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_cfg = copy.deepcopy(norm_cfg)
        super(ViPNASMobileNetV3, self).__init__()
        self.wid = wid
        self.expan = expan
        self.dep = dep
        self.ks = ks
        self.group = group
        self.att = att
        self.stride = stride
        self.act = act
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.wid[0],
            kernel_size=self.ks[0],
            stride=self.stride[0],
            padding=self.ks[0] // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type=self.act[0]))

        self.layers = self._make_layer()

    def _make_layer(self):
        """Define layers"""
        seq_layers = nn.CellList()
        for i, dep in enumerate(self.dep[1:]):
            mid_channels = self.wid[i + 1] * self.expan[i + 1]

            if self.att[i + 1]:
                se_cfg = dict(
                    channels=mid_channels,
                    ratio=4,
                    act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')))
            else:
                se_cfg = None

            if self.expan[i + 1] == 1:
                with_expand_conv = False
            else:
                with_expand_conv = True

            layer_list = []
            for j in range(dep):
                if j == 0:
                    stride = self.stride[i + 1]
                    in_channels = self.wid[i]
                else:
                    stride = 1
                    in_channels = self.wid[i + 1]

                layer = InvertedResidual(
                    in_channels=in_channels,
                    out_channels=self.wid[i + 1],
                    mid_channels=mid_channels,
                    kernel_size=self.ks[i + 1],
                    groups=self.group[i + 1],
                    stride=stride,
                    se_cfg=se_cfg,
                    with_expand_conv=with_expand_conv,
                    conv_cfg=self.conv_cfg,
                    act_cfg=dict(type=self.act[i + 1]))
                layer_list.append(layer)
            seq_layers = nn.CellList(layer_list)
        return seq_layers

    def construct(self, x):
        x = self.conv1.construct(x)

        for block in self.seq_layers:
            x = block(x)

        return x


class ConvModule(nn.Cell):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if not (conv_cfg is None or isinstance(conv_cfg, dict)):
            raise TypeError('conv_cfg should be None or dict')
        if not (norm_cfg is None or isinstance(norm_cfg, dict)):
            raise TypeError('norm_cfg should be None or dict')
        if not (act_cfg is None or isinstance(act_cfg, dict)):
            raise TypeError('act_cfg should be None or dict')
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = order
        if not (isinstance(self.order, tuple) and len(self.order) == 3):
            raise KeyError('order should be tuple and its length should be 3')
        if set(order) != {'conv', 'norm', 'act'}:
            raise TypeError('order should be the set of conv, norm and act ')

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        if padding == 0:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='valid',
                dilation=dilation,
                group=groups,
                has_bias=bias,
                weight_init=Normal(sigma=0.001),
                bias_init='zeros')
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                pad_mode='pad',
                dilation=dilation,
                group=groups,
                has_bias=bias,
                weight_init=Normal(sigma=0.001),
                bias_init='zeros')

        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = nn.BatchNorm2d(num_features=norm_channels)
        else:
            self.norm = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            self.activate = build_activation_layer(act_cfg_)
        else:
            self.activate = None

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        """Initial weights"""
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        constant_init(self.norm, 1, bias=0)

    def construct(self, x):
        """Construct conv blocks"""
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.norm is not None:
                x = self.norm(x)
            elif layer == 'act' and self.activate is not None:
                x = self.activate(x)
        return x


class InvertedResidual(nn.Cell):
    """Inverted Residual Block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Default: 3.
        groups (None or int): The group number of the depthwise convolution.
            Default: None, which means group number = mid_channels.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Default: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels.
            Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for norm layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: None.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 groups=None,
                 stride=1,
                 se_cfg=None,
                 with_expand_conv=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        # Protect mutable default arguments
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        act_cfg = copy.deepcopy(act_cfg)
        super(InvertedResidual, self).__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        if stride not in [1, 2]:
            raise KeyError('stride should be 1 or 2')
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv

        if groups is None:
            groups = mid_channels

        if self.with_se and not isinstance(se_cfg, dict):
            raise TypeError('the type of se_cfg should be dict')
        if not self.with_expand_conv and mid_channels != in_channels:
            raise KeyError('mid_channels should be equal to in_channels')

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def construct(self, x):
        """Construct Inverted Residual Blocks"""

        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv.construct(out)

            out = self.depthwise_conv.construct(out)

            if self.with_se:
                out = self.se.construct(out)

            out = self.linear_conv.construct(out)

            if self.with_res_shortcut:
                return x + out
            return out

        out = _inner_forward(x)

        return out


class SELayer(nn.Cell):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        self.global_avgpool = ops.AdaptiveAvgPool2D(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def construct(self, x):
        out = self.global_avgpool(x)
        out = self.conv1.construct(out)
        out = self.conv2.construct(out)
        return x * out


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


def build_activation_layer(act_cfg):
    """Build activation layer"""
    if act_cfg.get('type') == 'ReLU':
        return nn.ReLU()
    if act_cfg.get('type') == 'HSwish':
        return nn.HSwish()
    if act_cfg.get('type') == 'Sigmoid':
        return nn.Sigmoid()
    if act_cfg.get('type') == 'HSigmoid':
        return HSigmoid()
    return None


class HSigmoid(nn.Cell):
    """Hard Sigmoid Module. Apply the hard sigmoid function:
    Hsigmoid(x) = min(max((x + bias) / divisor, min_value), max_value)
    Default: Hsigmoid(x) = min(max((x + 1) / 2, 0), 1)

    Args:
        bias (float): Bias of the input feature map. Default: 1.0.
        divisor (float): Divisor of the input feature map. Default: 2.0.
        min_value (float): Lower bound value. Default: 0.0.
        max_value (float): Upper bound value. Default: 1.0.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def construct(self, x):
        x = (x + self.bias) / self.divisor
        return ops.clip_by_value(x, self.min_value, self.max_value)
