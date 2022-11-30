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
""" Style transfer network."""

from mindspore import nn, ops, Tensor

class BasicConv2d(nn.Cell):
    """
    Convolution operation followed by optional batch_norm and activation function.

    Args:
        in_channels (int): Input channels of convolution.
        out_channels (int): Output channels of convolution.
        kernel_size (int): Shape of convolution kernel, expected to be odd. Default: 3.
        stride (int): Stride of convolution. Default: 1.
        activation_fn (nn.Cell): Activation operation after convolution. Default: nn.ReLU().
        normalizer_fn (nn.Cell): Batch_norm operation after convolution. Default: None.
        pad_mode (str): Padding mode of convolution. Optional values are "same", "valid", "pad", Default: "same".

    Inputs:
        -**x** (Tuple) - A tuple including a tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`, a customized
        batch_norm operation and a dict of normalization parameters.

    Outputs:
        Tuple of tensor, nn.Cell, dict and int.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> block = BasicConv2d(3, 10)
        >>> x = Tensor(np.ones([1, 3, 128, 128]), mindspore.float32)
        >>> output = block((x, None, None, 0))
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, activation_fn=nn.ReLU(),
                 normalizer_fn=None, pad_mode='same', **kwargs):
        super(BasicConv2d, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, **kwargs)
        self.bn = normalizer_fn
        self.activation_fn = activation_fn
        self.pad = ops.MirrorPad(mode='REFLECT')
        self.paddings = Tensor([[0, 0], [0, 0], [padding, padding], [padding, padding]])

    def construct(self, x):
        x, normalizer_fn, params, order = x
        x = self.conv(x)
        if normalizer_fn:
            x = normalizer_fn((x, params, order))
        if self.bn:
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return (x, normalizer_fn, params, order + 1)

class Residual(nn.Cell):
    """
    A residual block with contional normalization.

    Args:
        channels (int): number of input and output channels.
        kernel_size (int): an odd number representing the kernel size.

    Inputs:
        -**x** (Tuple) - A tuple including a tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`, a customized
        batch_norm operation and a dict of normalization parameters.

    Outputs:
        Tuple of tensor, nn.Cell, dict and int.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> res = Residual(10, 3)
        >>> x = Tensor(np.ones([1, 10, 128, 128]), mindspore.float32)
        >>> output = block((x, None, None, 0))
    """
    def __init__(self, channels, kernel_size):
        super(Residual, self).__init__()
        self.conv1 = BasicConv2d(channels, channels, kernel_size=kernel_size, stride=1)
        self.conv2 = BasicConv2d(channels, channels, kernel_size=kernel_size, stride=1, activation_fn=None)

    def construct(self, x):
        h_1 = self.conv1(x)
        h_2 = self.conv2(h_1)
        out1, _, _, _ = x
        out2, normalizer_fn, params, order = h_2
        return (out1 + out2, normalizer_fn, params, order)

class Upsampling(nn.Cell):
    """
    Computes a nearest-neighbor upsampling of the input by a factor of 'stride',
    then applies convolution and conditional normalization.

    Args:
        stride (int): Multiple of enlarge operation.
        size (int): Init size of input.
        kernel_size (int): Size of convolution kernel.
        in_channels (int): Input channels of convolution.
        out_channels (int): Output channels of convolution.
        activation_fn (nn.Cell): Activation after convolution. Default: nn.ReLU().

    Inputs:
        -**input_** (Tuple) - A tuple including a tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`, a customized
        batch_norm operation and a dict of normalization parameters.

    Outputs:
        Tuple of tensor, nn.Cell, dict and int.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> upsample = Upsampling(2, (128, 128), 3, 10, 64)
        >>> x = Tensor(np.ones([1, 10, 128, 128]), mindspore.float32)
        >>> output = upsample((x, None, None, 0))
    """
    def __init__(self, stride, size, kernel_size, in_channels, out_channels, activation_fn=nn.ReLU()):
        super().__init__()
        self.stride = stride
        self.resize = ops.ResizeNearestNeighbor([i*stride for i in size])
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size=kernel_size, activation_fn=activation_fn)
    def construct(self, input_):
        x, normalizer_fn, params, order = input_
        _, _, height, width = x.shape
        resize = ops.ResizeNearestNeighbor([height * self.stride, width * self.stride])
        x = resize(x)
        x = self.conv((x, normalizer_fn, params, order))
        return x


class Transform(nn.Cell):
    """
    Map content images to stylized images with style embedding.

    Args:
        in_channel (int): number of input channels. Default: 3.
        alpha (float): width multiplier to reduce the number of filters
            used in the model and slim it down. Default: 1.0.
    Inputs:
        -**x** (Tuple) - A tuple including a tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`, a customized
        batch_norm operation and a dict of normalization parameters.

    Outputs:
        Tensor of shape : math:`(N, 3, H_{out}, W_{out}).

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> trans = Transform()
        >>> content = Tensor(np.ones([1, 3, 128, 128]), mindspore.float32)
        >>> output = trans((content, None, None))
    """
    def __init__(self, in_channels=3, alpha=1.0):
        super(Transform, self).__init__()
        self.contract = nn.SequentialCell([
            BasicConv2d(in_channels, int(alpha * 32), kernel_size=9, stride=1,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 32), eps=0.001)),
            BasicConv2d(int(alpha * 32), int(alpha * 64), kernel_size=3, stride=2,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 64), eps=0.001)),
            BasicConv2d(int(alpha * 64), int(alpha * 128), kernel_size=3, stride=2,
                        normalizer_fn=nn.BatchNorm2d(int(alpha * 128), eps=0.001))
        ])
        self.residual = nn.SequentialCell([
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3),
            Residual(int(alpha * 128), 3)
            ])
        self.expand = nn.SequentialCell([
            Upsampling(2, (32, 32), 3, int(alpha * 128), int(alpha * 64)),
            Upsampling(2, (64, 64), 3, int(alpha * 64), int(alpha * 32)),
            Upsampling(1, (128, 128), 9, int(alpha * 32), 3, activation_fn=nn.Sigmoid())
        ])

    def construct(self, x):
        x, normalizer_fn, style_params = x
        out = self.contract((x, None, None, 0))
        x, _, _, _ = out
        x = self.residual((x, normalizer_fn, style_params, 0))
        out = self.expand(x)
        x, _, _, _ = out
        return x

class ConditionalStyleNorm(nn.Cell):
    """
    Conditional style normalization.

    Used as a normalization function for conv2d with specific parameters.

    Args:
        style_params (dict): A dict from the scope names of the variables of this method + beta/gamma
            to the beta and gamma tensors. Default: None.
            eg. {'transformer/expand/conv2/conv/StyleNorm/beta': <tf.Tensor>,
            'transformer/expand/conv2/conv/StyleNorm/gamma': <tf.Tensor>,
            'transformer/residual/residual1/conv1/StyleNorm/beta': <tf.Tensor>,
            'transformer/residual/residual1/conv1/StyleNorm/gamma': <tf.Tensor>}
        activation_fn (function): Optional activation function. Default: None.

    Inputs:
        -**input_** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape : math:`(N, C_{out}, H_{out}, W_{out}).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> norm = ConditionalStyleNorm()
    """
    def __init__(self, style_params=None, activation_fn=None):
        super(ConditionalStyleNorm, self).__init__()
        self.style_params = style_params
        self.moments = nn.Moments(axis=(2, 3), keep_dims=True)
        self.activation_fn = activation_fn
        self.rsqrt = ops.Rsqrt()
        self.cast = ops.Cast()

    def get_style_parameters(self, style_params):
        """Gets style normalization parameters."""
        var = []
        for i in style_params.keys():
            var.append(style_params[i].expand_dims(2).expand_dims(3))
        return var

    def norm(self, x, mean, variance, style_parameters, variance_epsilon, order):
        """ Normalization function with specific parameters. """
        inv = self.rsqrt(variance + variance_epsilon)
        gamma = style_parameters[order*2+1]
        beta = style_parameters[order*2]
        if gamma is not None:
            inv *= gamma
        data1 = self.cast(inv, x.dtype)
        data2 = x * data1
        data3 = mean * inv
        if gamma is not None:
            data4 = beta - data3
        else:
            data4 = -data3
        data5 = data2 + data4
        return data5

    def construct(self, input_):
        x, style_params, order = input_
        mean, variance = self.moments(x)
        style_parameters = self.get_style_parameters(style_params)
        output = self.norm(x, mean, variance, style_parameters, 1e-5, order)
        if self.activation_fn:
            output = self.activation_fn(output)
        return output
