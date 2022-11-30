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
"""
Custom ops for efficient bias and activation.
"""

from mindspore import nn, ops, Tensor


activation_funcs = dict()
activation_funcs['linear'] = dict(def_alpha=0, def_gain=1)
activation_funcs['lrelu'] = dict(def_alpha=0.2, def_gain=1.4142)


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None):
    """Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x: Input activation tensor. Can be of any shape.
        b: Bias vector, or `None` to disable. Must be a 1D tensor of the same type as `x`.
            The shape must be known, and it must match the dimension of `x` corresponding to `dim`. Default: None.
        dim: The dimension in `x` corresponding to the elements of `b`.
            The value of `dim` is ignored if `b` is not specified. Default: 1.
        act: Name of the activation function to evaluate, or `"linear"` to disable.
            Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc. Default: 'linear'.
        alpha: Shape parameter for the activation function, or `None` to use the default. Default: None.
        gain: Scaling factor for the output tensor. Default: None.

    Returns:
        Tensor of the same shape and datatype as `x`.

    Examples:
        >>> x = bias_act(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain)
    """

    spec = activation_funcs[act]
    alpha = alpha if alpha is not None else spec['def_alpha']
    gain = gain if gain is not None else spec['def_gain']

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
    clip_min = Tensor(-255, x_type)
    clip_max = Tensor(255, x_type)
    x = ops.clip_by_value(x, clip_min, clip_max)
    return x
