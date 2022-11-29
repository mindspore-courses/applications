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
"""Custom ops for efficient bias and activation."""

from mindspore import nn, ops, Tensor


activation_funcs = dict()
activation_funcs['linear'] = {'def_alpha': 0, 'def_gain': 1}
activation_funcs['lrelu'] = {'def_alpha': 0.2, 'def_gain': 1.414}


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """
    Slow reference implementation of `bias_act()` using standard TensorFlow ops. Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x (Tensor): Input activation tensor. Can be of any shape.
        b (Tensor): Bias vector, or `None` to disable. Must be a 1D tensor of the same type as `x`. Default: None.
        dim (int): The dimension in `x` corresponding to the elements of `b`. Default: 1.
        act (str): Name of the activation function to evaluate, can be `linear` or `lrelu`. Default: ‘linear’.
        alpha (float): Shape parameter for the activation function. Default: None.
        gain (int): Scaling factor for the output tensor, or `None` to use default. Default: None.
        clamp (int): Clamp the output values to `[-clamp, +clamp]`, or `None` to disable the clamping. Default: None.

    Returns:
        Tensor, output tensor of the same shape and datatype as `x`.

    Examples:
        >>> x = bias_act(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)
    """
    spec = activation_funcs[act]
    alpha = alpha if alpha is not None else spec['def_alpha']
    gain = gain if gain is not None else spec['def_gain']
    _ = clamp if clamp is not None else -1

    # Add bias.
    if b is not None:
        new_shape = []
        for i in range(x.ndim):
            if i == dim:
                new_shape.append(-1)
            else:
                new_shape.append(1)
        x = x + b.reshape(new_shape)

    # Evaluate activation function.
    if act == 'lrelu':
        x = nn.LeakyReLU(alpha)(x)

    # Scale by gain.
    if gain != 1:
        x = x * gain

    # Clamp.
    x_type = x.dtype
    clip_max = Tensor(255, x_type)
    clip_min = Tensor(-255, x_type)
    x = ops.clip_by_value(x, clip_min, clip_max)
    return x
