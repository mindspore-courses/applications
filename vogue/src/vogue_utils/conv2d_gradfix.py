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
"""Custom replacement for conv2d that supports
arbitrarily high order gradients with zero performance penalty."""

from mindspore import load_param_into_net


def find_func(xx, ww, name, all_info):
    """
    Find func in all_info.

    Args:
        xx (Tensor): Input.
        ww (Tensor): Weight.
        name (str): Func name, in ['conv2d', 'transpose2d'].
        all_info (list): Information of all_conv. Default: None.

    Returns:
        nn.Cell, the chosen func.
        Tensor, the given weight.

    Examples:
        >>> func, ww = find_func(x_input, weight, 'conv2d', all_info)
    """
    all_conv, all_conv_weight, input_list, weight_list, name_list = all_info
    xx_shape = xx.shape
    ww_shape = ww.shape
    for ii, (conv_one, input_shape_one, weight_shape_one, name_one) in \
            enumerate(zip(all_conv, input_list, weight_list, name_list)):
        if xx_shape == input_shape_one and ww_shape == weight_shape_one and name == name_one:
            func = conv_one
            conv_weight = func.parameters_dict()
            for weight_name in conv_weight.keys():
                conv_weight[weight_name].set_data(ww)
                load_param_into_net(func, conv_weight)
            return func, all_conv_weight[ii]
    return None, None


def conv2d(x_input, weight, all_info=None):
    """
    Conv2d operation.

    Args:
        x_input (Tensor): Input.
        weight (Tensor): Weight.
        all_info (list): Information of all_conv. Default: None.

    Returns:
        Tensor, output tensor after conv2d.

    Examples:
        >>> x = conv2d(x_input, weight, all_info=all_info)
    """
    func, ww = find_func(x_input, weight, 'conv2d', all_info)
    print(ww[0][0][0])
    out = func(x_input)
    return out


def conv_transpose2d(x_input, weight, all_info=None):
    """
    Transpose conv2d operation.

    Args:
        x_input (Tensor): Input.
        weight (Tensor): Weight.
        all_info (list): Information of all_conv. Default: None.

    Returns:
        Tensor, output tensor after conv_transpose2d.

    Examples:
        >>> x = conv_transpose2d(x_input, weight, all_info=all_info)
    """
    func, ww = find_func(x_input, weight, 'transpose2d', all_info)
    print(ww[0][0][0])
    out = func(x_input)
    return out
