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
"""2D convolution with optional up/downsampling."""

from . import conv2d_gradfix
from . import upfirdn2d


def _get_weight_shape(w):
    """
    Get the shape of the weight.

    Args:
        w (Tensor): The weight.

    Returns:
        list, the shape.

    Examples:
          >>> shape = _get_weight_shape(w)
    """
    shape = [sz for sz in w.shape]
    return shape


def conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True, all_info=None):
    """
    Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.

    Args:
        x (Tensor): Input.
        w (Tensor): Weight.
        stride (int): Stride. Default: 1.
        padding (int): Padding. Default: 0.
        groups (int): Groups. Default: 1.
        transpose (bool): Need to transpose. Default: False.
        flip_weight (bool): Need to flip the weight. Default: True.
        all_info (list): Information of all_conv. Default: None.

    Returns:
        Tensor, output of conv2d.

    Examples:
        >>> x = conv2d_wrapper(x=x, w=w, groups=groups, transpose=True, flip_weight=True, all_info=all_info)
    """
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)

    # Flip weight if requested.
    if not flip_weight:
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if kw == 1 and kh == 1 and stride == 1 and padding in [0, [0, 0], (0, 0)] and not transpose:
        if x.strides[1] == 1:
            if min(out_channels, in_channels_per_group) < 64:
                if out_channels <= 4 and groups == 1:
                    in_shape = x.shape
                    x = w.squeeze(3).squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                    x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
                else:
                    raise AssertionError('shape of x is nor correct')
    # Otherwise => execute using conv2d_gradfix.
    op = conv2d_gradfix.conv_transpose2d if transpose else conv2d_gradfix.conv2d
    output = op(x, w, all_info=all_info)
    return output


def conv2d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False,
                    all_info=None):
    """
    2D convolution with optional up/downsampling.

    Padding is performed only once at the beginning, not between the operations.

    Args:
        x (Tensor): Input tensor of shape `[batch_size, in_channels, in_height, in_width]`.
        w (Tensor): Weight tensor of shape `[out_channels, in_channels//groups, kernel_height, kernel_width]`.
        f (Tensor): Low-pass filter for up/downsampling. Must be prepared beforehand by
            calling upfirdn2d.setup_filter(). None = identity. Default: None.
        up (int): Integer upsampling factor. Default: 1.
        down (int): Integer downsampling factor. Default: 1.
        padding (int): Padding with respect to the upsampled image. Can be a single number
            or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`. Default: 0.
        groups (int): Split input channels into N groups. Default: 1.
        flip_weight (bool): False = convolution, True = correlation. Default: True.
        flip_filter (bool): False = convolution, True = correlation. Default: False.
        all_info (list): information of all_conv. Default: None.

    Returns:
        Tensor, tensor of the shape `[batch_size, num_channels, out_height, out_width]`.

    Examples:
        >>> x = conv2d_resample(x=x, w=w, f=f, flip_weight=flip_weight, all_info=all_info)
    """
    # Validate arguments.
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    fw, fh = upfirdn2d.get_filter_size(f)
    px0, px1, py0, py1 = upfirdn2d.parse_padding(padding)

    # Adjust padding to account for up/downsampling
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    if kw == 1 and kh == 1:
        # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
        if down > 1 and up == 1:
            x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter,
                                    all_info=all_info)
            x = conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight, all_info=all_info)
            return x

        # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
        if up > 1 and down == 1:
            x = conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight, all_info=all_info)
            x = upfirdn2d.upfirdn2d(x=x, f=f, up=up, padding=[px0, px1, py0, py1], gain=up ** 2,
                                    flip_filter=flip_filter, all_info=all_info)
            return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, padding=[px0, px1, py0, py1], flip_filter=flip_filter, all_info=all_info)
        x = conv2d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight, all_info=all_info)
        return x

    if up > 1:
        # Fast path: upsampling with optional downsampling => use transpose strided convolution. (if up > 1)
        if groups == 1:
            w = w.transpose(1, 0, 2, 3)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(0, 2, 1, 3, 4)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = conv2d_wrapper(x=x, w=w, stride=up, padding=pxt, groups=groups, transpose=True,
                           flip_weight=(not flip_weight), all_info=all_info)
        x = upfirdn2d.upfirdn2d(x=x, f=f, padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt], gain=up ** 2,
                                flip_filter=flip_filter, all_info=all_info)
        if down > 1:
            x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter, all_info=all_info)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return conv2d_wrapper(x=x, w=w, padding=px0, groups=groups, flip_weight=flip_weight, all_info=all_info)

    # Fallback: Generic reference implementation.
    x = upfirdn2d.upfirdn2d(x=x, f=(f if up > 1 else None), up=up, padding=[px0, px1, py0, py1],
                            gain=up**2, flip_filter=flip_filter, all_info=all_info)
    x = conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight, all_info=all_info)
    if down > 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter, all_info=all_info)
    return x
