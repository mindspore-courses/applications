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

"""Some functions used in constructing models"""

import mindspore as ms
import mindspore.nn as nn


def adaptive_avg_pool_2d_func_gpu(inputs, output_res):
    """
    Adaptive average pooling function on GPU.

    Args:
        inputs (Tensor): Inputs.
        output_res (list): The resolution of the output.

    Returns:
        Tensor, pooled result.
    """
    if isinstance(output_res, int):
        output_res = [output_res, output_res]
    pooler = ms.ops.AdaptiveAvgPool2D(tuple(output_res))
    return pooler(inputs)

class UpsampleNearest(nn.Cell):
    """
    Nearest neighborhood upsampling function.

    Args:
        scale_factor (int): The scaling factor of inputs.

    Inputs:
        -**features** (Tensor) - 4-dim features (batch_size, channel_num, height, width).

    Outputs:
        -**scaled features** (Tensor) - The scaled output.
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def construct(self, inputs):
        """construct"""
        output_res = [inputs.shape[-2]*self.scale_factor, inputs.shape[-1]*self.scale_factor]
        resizer = ms.ops.ResizeNearestNeighbor(output_res)
        return resizer(inputs)

def channel_shuffle(inputs, groups):
    """
    Shuffle the channels in the inputs by the transpose operation.

    Args:
        inputs (Tensor): Inputs.
        groups (int): The number of groups.

    Result:
        Tensor, shuffled result.
    """
    batch_size, num_channels, height, width = inputs.shape
    channels_per = num_channels // groups
    inputs = inputs.reshape(batch_size, groups, channels_per, height, width)
    inputs = inputs.transpose(0, 2, 1, 3, 4)
    inputs = inputs.reshape(batch_size, -1, height, width)
    return inputs


def channel_split(inputs, groups, axis=1):
    """
    Split the channels into groups.

    Args:
        inputs (Tensor): Inputs.
        groups (int): The number of output groups.
        axis (int): The axis to be split. Default: 1.

    Returns:
        Tensor, split result.
    """
    output = []
    if isinstance(groups, int):
        for i in ms.numpy.split(inputs, indices_or_sections=groups, axis=axis):
            output.append(i)

    else:
        former_index = 0
        output = []
        for i in groups:
            if axis == 1:
                output.append(inputs[:, former_index:former_index+i])
            else:
                output.append(inputs[former_index:former_index+i])

            former_index = former_index+i

    return output

class IdentityMap(nn.Cell):
    """
    Identity mapping.

    Returns:
        Tensor, input itself.
    """
    def construct(self, x):
        """construct"""
        return x
