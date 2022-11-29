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
"""Common block for CAR model"""

import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class PixelUnShuffle(nn.Cell):
    r"""
    Rearranges elements in a tensor of shape :math:`(*, C, H \times r, W \times r)`
    to a tensor of shape :math:`(*, C \times r^2, H, W)`, where r is an downscale factor.

    Args:
        scale(int): The downscale factor.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, The tensor after unshuffle.
    """

    def __init__(self, scale):
        super().__init__()
        self.down_scale = scale
    def construct(self, x):
        b, c, h, w = x.shape
        oc = c * self.down_scale ** 2
        oh = int(h / self.down_scale)
        ow = int(w / self.down_scale)
        output_reshaped = x.reshape(b, c, oh, self.down_scale, ow, self.down_scale)
        output = output_reshaped.transpose(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)

        return output


class PixelShuffle(nn.Cell):
    r"""
    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.

    Args:
        scale(int): The upscale factor.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after shuffle.
    """

    def __init__(self, scale):
        super().__init__()
        self.down_scale = scale
    def construct(self, x):
        b, c, h, w = x.shape
        oc = c // (self.down_scale ** 2)
        oh = h * self.down_scale
        ow = w * self.down_scale
        output_reshaped = x.reshape(b, oc, self.down_scale, self.down_scale, h, w)
        output = output_reshaped.transpose(0, 1, 4, 2, 5, 3).reshape(b, oc, oh, ow)

        return output


class ReflectionPad2d(nn.Cell):
    """
    Pads the input tensor using the reflection of the input boundary.

    Args:
        padding (Union[int, tuple]): the size of the padding. If is `int`, uses the same
            padding in all boundaries.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `paddings` is not a tuple or int.
        ValueError: If length of `paddings` is more than 4 or its shape is not (N, 2).
    """

    def __init__(self, paddings):
        super().__init__()
        if isinstance(paddings, int):
            paddings = ((0, 0), (0, 0), (paddings, paddings), (paddings, paddings))
        if not isinstance(paddings, tuple):
            raise TypeError(
                f"For '{self.cls_name}', the type of 'paddings' must be tuple, "
                f"but got {type(paddings).__name__}."
            )
        for item in paddings:
            if len(item) != 2:
                raise ValueError(
                    f"For '{self.cls_name}', the dimension of 'paddings' must be (n, 2), "
                    f"but got {paddings}."
                )
        if len(paddings) > 4:
            raise ValueError(
                f"For '{self.cls_name}', only 'paddings' up to 4 dims is supported, but got "
                f"{len(paddings)}."
            )
        self.ops = nn.Pad(paddings, mode="REFLECT")

    def construct(self, x):
        return self.ops(x)


class DownsampleBlock(nn.Cell):
    """
    Combine the PixelUnShuffle and Conv2d.

    Args:
        scale(int): The upscale factor.
        input_channels (int): Input feature dimensions.
        output_channels(int): Output feature dimensions.
        ksize (int): downsample kernel size. Default:1.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after downsample.
    """

    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super().__init__()
        self.downsample = nn.SequentialCell(
            PixelUnShuffle(scale),
            nn.Conv2d(
                input_channels * (scale**2),
                output_channels,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
                pad_mode="valid",
                has_bias=True
            )
        )
    def construct(self, x):
        return self.downsample(x)


class UpsampleBlock(nn.Cell):
    """
    Combine the Conv2d and PixelShuffle.

    Args:
        scale(int): The upscale factor.
        input_channels (int): Input feature dimensions.
        output_channels(int): Output feature dimensions.
        ksize (int): downsample kernel size. Default:1.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after downsample.
    """

    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super().__init__()
        self.upsample = nn.SequentialCell(
            nn.Conv2d(
                input_channels,
                output_channels * (scale**2),
                kernel_size=1,
                stride=1,
                padding=ksize // 2,
                pad_mode="pad",
                has_bias=True
            ),
            PixelShuffle(scale)
        )

    def construct(self, x):
        return self.upsample(x)


class NormalizeBySum(nn.Cell):
    """
    Normalize feature channel.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after normalize.
    """

    def __init__(self):
        super().__init__()
        self.min_value = mindspore.Tensor(1e-6, mindspore.float32)
    def construct(self, x):
        sum_x = x.sum(axis=1, keepdims=True)
        sum_x = ops.clip_by_value(sum_x, self.min_value)
        x = x / sum_x
        return x


class MeanShift(nn.Conv2d):
    """
    Shift the mean RGB value.

    Args:
        rgb_range (int): Indicate the value range. default: 1.
        rgb_mean (tuple):  RGB mean value. Default: (0.4488, 0.4371, 0.4040).
        rgb_std(tuple): RGB std value.  Default: (1.0, 1.0, 1.0).
        sign (int): Add or subtracted. default: -1.
    """
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1, has_bias=True)
        std = mindspore.Tensor(rgb_std)
        self.weight.set_data(ops.Eye()(3, 3, mindspore.float32).view(3, 3, 1, 1) / std.view(3, 1, 1, 1))
        self.bias.set_data(sign * rgb_range * mindspore.Tensor(rgb_mean) / std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class ResidualBlock(nn.Cell):
    """
    Build the residual module.

    Args:
        input_channels (int): input channel.
        channels (int):  output channel.
        ksize(int): conv kernel size. Default:3.
        use_instance_norm (bool): A bool value, determined if insert InstanceNorm2d. Default:False.
        affine (bool): A bool value. For InstanceNorm2d. Default: False.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor.
    """

    def __init__(self, input_channels, channels, ksize=3, use_instance_norm=False, affine=False):
        super().__init__()
        self.channels = channels
        self.ksize = ksize
        padding = self.ksize // 2
        if use_instance_norm:
            self.transform = nn.SequentialCell(
                ReflectionPad2d(padding),
                nn.Conv2d(input_channels, channels, kernel_size=self.ksize,
                          stride=1, pad_mode="valid", has_bias=True),
                nn.InstanceNorm2d(channels, affine=affine),
                nn.LeakyReLU(0.2),
                ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size=self.ksize,
                          stride=1, pad_mode="valid", has_bias=True),
                nn.InstanceNorm2d(channels),
            )
        else:
            self.transform = nn.SequentialCell(
                ReflectionPad2d(padding),
                nn.Conv2d(input_channels, channels, kernel_size=self.ksize,
                          stride=1, pad_mode="valid", has_bias=True),
                nn.LeakyReLU(0.2),
                ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size=self.ksize,
                          stride=1, pad_mode="valid", has_bias=True),
            )

    def construct(self, x):
        return x + self.transform(x)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """
    Warp a standard convolution layer.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (int): Specifies the height and width of the 2D convolution kernel.
        bias (bool): Whether the Conv2d layer has a bias parameter. Default: False.

    Returns:
        Conv2d.
    """

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), pad_mode="pad", has_bias=bias)


class ResBlock(nn.Cell):
    """
    The residual module for super resolution model.

    Args:
        conv (Cell): The convolution layer.
        n_feats(int): Input feature dimensions.
        kernel_size(int): Whether add bacth_norm layer. Default:False.
        bias(bool): Whether add bias, Default:True.
        bn(bool): Whether add bacth_norm layer. Default:False.
        act(Cell): Activation layer. Default:nn.ReLU.
        res_scale(float): Scaling the result. Default:1.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor.
    """

    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(*m)
        self.res_scale = res_scale

    def construct(self, x):
        res = self.body(x) * (self.res_scale)
        res += x
        return res


class Upsampler(nn.SequentialCell):
    """
    Upsampler Block for super resolution model.

    Args:
        conv (Cell): The convolution layer.
        scale(int): The upscaling rate.
        n_feats(int): Input feature dimensions.
        bn(bool): whether add bacth_norm layer. Default:False
        act(bool): whether add activation layer. Default:False
        bias(bool): Whether add bias for convolution. Default:True
    """

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU())
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Quantization(nn.Cell):
    """
    Define quantify operation and backpropagation.

    Inputs:
        - **img** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after quantized.
    """

    def __init__(self):
        super().__init__()
        self.round = ops.Round()
        self.clip = ops.clip_by_value
        self.cos = ops.Cos()
        self.pi = 3.1415926
        self.alp = 1.

    def construct(self, img):
        img = self.clip(img, 0, 1)
        img = img * 255
        img = self.round(img)
        return img / 255

    def bprop(self, img, out, grad_output):
        _ = out

        grad_input = grad_output
        grad_input = grad_output*(1-self.alp*self.cos(2*self.pi*img))
        return (grad_input,)
