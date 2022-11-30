# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://ww.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Custom replacement for conv2d that supports arbitrarily high order gradients with zero performance penalty.
"""

import mindspore as ms


def get_func(x_input, weight, name_conv, conv_info):
    """
    Retrieve func in conv_info.

    Args:
        x_input (Tensor): Input.
        weight (Tensor): Weight.
        name_conv (str): conv name, option = ['conv2d', 'transpose2d'].
        conv_info (list): Information of conv_list. Default: None.

    Returns:
        Cell, the chosen func.
        Tensor, the given weight.

    Examples:
        >>> func, w = get_func(x_input, weight, 'conv2d', conv_info)
    """
    conv_list, conv_list_weight, input_list, weight_list, name_list = conv_info
    for ii, (conv, input_shape, weight_shape, name) in enumerate(zip(conv_list, input_list, weight_list, name_list)):
        if x_input.shape == input_shape and weight.shape == weight_shape and name_conv == name:
            func = conv
            ms.ops.Assign()(conv_list_weight[ii], weight)
            return func, conv_list_weight[ii]
    return None, None


def conv2d(x_input, weight, conv_info=None):
    """
    Conv2d operation.

    Args:
        x_input (Tensor): Input.
        weight (Tensor): Weight.
        conv_info (list): Information of conv_list. Default: None.

    Returns:
        Tensor, output tensor of conv2d.

    Examples:
        >>> x = conv2d(x_input, weight, conv_info=conv_info)
    """
    conv, w = get_func(x_input, weight, 'conv2d', conv_info)
    print(w[0][0][0][0])
    out = conv(x_input)
    return out


def conv_transpose2d(x_input, weight, conv_info=None):
    """
    Transpose conv2d operation.

    Args:
        x_input (Tensor): Input.
        weight (Tensor): Weight.
        conv_info (list): Information of conv_list. Default: None.

    Returns:
        Tensor, output tensor of conv_transpose2d.

    Examples:
        >>> x = conv_transpose2d(x_input, weight, conv_info=conv_info)
    """
    conv, w = get_func(x_input, weight, 'transpose2d', conv_info)
    print(w[0][0][0][0])
    out = conv(x_input)
    return out
