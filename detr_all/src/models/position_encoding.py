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
# ======================================================================
"""position embedding"""
import math

import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.tensor import Tensor


class PositionEmbeddingSine(nn.Cell):
    """
    Position embedding architecture.

    Args:
        pos_dim (int): Position feature dimension.
        normalize (bool): Normalize the position embedding if true.
        scale (float):  Scale the position embedding.

    Returns:
        pos (Cell): cell instance of position embedding.

    Inputs:
    - **mask** (Tensor) - Image mask with shape (N,H,W).

    Outputs:
        Tensor, with shape (N,256,H,W). Position embedding Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> p = PositionEmbeddingSine()
    """

    def __init__(self, pos_dim=128, normalize=True, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.pos_dim = pos_dim
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, mask):
        """Apply Position embedding architecture"""
        mask = 1 - mask
        y_embed = mask.cumsum(1)
        x_embed = mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = Tensor(np.arange(self.pos_dim).astype(np.float32))
        dim_t = ops.Pow()(10000, 2 * (dim_t // 2) / self.pos_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        sin = ops.Sin()
        cos = ops.Cos()
        stack = ops.Stack(axis=4)
        pos_x = stack((sin(pos_x[:, :, :, 0::2]), cos(pos_x[:, :, :, 1::2])))
        new_shape = (pos_x.shape[0], pos_x.shape[1],
                     pos_x.shape[2], pos_x.shape[3] * pos_x.shape[4])
        reshape = ops.Reshape()
        pos_x = reshape(pos_x, new_shape)
        pos_y = stack((sin(pos_y[:, :, :, 0::2]), cos(pos_y[:, :, :, 1::2])))
        pos_y = reshape(pos_y, new_shape)
        concatenate = ops.Concat(axis=3)
        pos = concatenate((pos_y, pos_x))
        transpose = ops.Transpose()
        pos = transpose(pos, (0, 3, 1, 2))
        return pos
