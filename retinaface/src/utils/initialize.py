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
"""Utils used to initialize tensors."""

import math
from functools import reduce

import numpy as np
from mindspore import Tensor


def init_kaiming_uniform(arr_shape, a=0, nonlinearity='leaky_relu', has_bias=False):
    """
    Kaiming initialize, generate a tensor with input shape, according to He initialization, using a uniform
    distribution.

    Args:
        arr_shape (tuple): The shape of generated tensor.
        a (float): Only use to leaky_relu, decide its' negative slope.
        nonlinearity (str): Non linearity function to be used, suggest to use relu or leaky_relu.
        has_bias (bool): Whether generate bias.

    Returns:
        A tuple, its first element is generated tuple with input shape, its second element is generated bias.

    """

    def _calculate_in_and_out(arr_shape):
        """Calculate input and output dimension of layer."""
        dim = len(arr_shape)
        if dim < 2:
            raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

        n_in = arr_shape[1]
        n_out = arr_shape[0]

        if dim > 2:
            counter = reduce(lambda x, y: x * y, arr_shape[2:])
            n_in *= counter
            n_out *= counter
        return n_in, n_out

    def calculate_gain(nonlinearity, a=None):
        """Calculate gain of Kaiming initialization."""
        linear_fans = ['linear', 'conv1d', 'conv2d', 'conv3d',
                       'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
        if nonlinearity in linear_fans or nonlinearity == 'sigmoid':
            return 1
        if nonlinearity == 'tanh':
            return 5.0 / 3
        if nonlinearity == 'relu':
            return math.sqrt(2.0)
        if nonlinearity == 'leaky_relu':
            if a is None:
                negative_slope = 0.01
            elif not isinstance(a, bool) and isinstance(a, int) or isinstance(a, float):
                negative_slope = a
            else:
                raise ValueError("negative_slope {} not a valid number".format(a))
            return math.sqrt(2.0 / (1 + negative_slope ** 2))

        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    fan_in, _ = _calculate_in_and_out(arr_shape)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    weight = np.random.uniform(-bound, bound, arr_shape).astype(np.float32)

    bias = None
    if has_bias:
        bound_bias = 1 / math.sqrt(fan_in)
        bias = np.random.uniform(-bound_bias, bound_bias, arr_shape[0:1]).astype(np.float32)
        bias = Tensor(bias)

    return Tensor(weight), bias
