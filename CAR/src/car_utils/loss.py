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
"""losses for car"""

import mindspore as ms
from mindspore import nn, ops

__all__ = ["TvLoss", "OffsetLoss"]

class OffsetLoss(nn.Cell):
    """
    Restrain the offset of resampling kernel.

    Args:
        kernel_size(int): The resampling kernel size.
        offsetloss_weight(float): To control the strength of loss.

    Inputs:
        - **offsets_h** (Tensor) - The horizontal shift of kernel.
        - **offsets_v** (Tensor) - The vertical shift of kernel.

    Returns:
        loss (float): Offset loss.
    """

    def __init__(self, kernel_size=13, offsetloss_weight=1.):
        super(OffsetLoss, self).__init__()
        self.offsetloss_weight = offsetloss_weight
        x = ms.numpy.arange(0, kernel_size, dtype=ms.float32)
        y = ms.numpy.arange(0, kernel_size, dtype=ms.float32)
        x_m, y_m = ops.Meshgrid()((x, y))
        self.sqrt = ops.Sqrt()
        weight = self.sqrt((x_m-kernel_size/2)**2 + (y_m-kernel_size/2)**2)/kernel_size
        self.weight = weight.view(1, kernel_size**2, 1, 1)

    def construct(self, offsets_h, offsets_v):
        b, _, h, w = offsets_h.shape
        loss = self.sqrt(offsets_h * offsets_h + offsets_v * offsets_v)*self.weight
        return self.offsetloss_weight*loss.sum()/(h * w * b)


class TvLoss(nn.Cell):
    """
    Restrain the movement of spatially neighboring resampling kernels.

    Args:
        tvloss_weight(float): To control the strength of loss.

    Inputs:
        - **offsets_h** (Tensor) - The horizontal shift of kernel.
        - **offsets_v** (Tensor) - The vertical shift of kernel.
        - **kernel** (Tensor) - The convolution kernel.

    Returns:
        loss (float): TV loss.
    """

    def __init__(self, tvloss_weight=1):
        super(TvLoss, self).__init__()
        self.tvloss_weight = tvloss_weight
        self.abs = ops.Abs()

    def construct(self, offsets_h, offsets_v, kernel):
        batch, _, _, _ = offsets_h.shape
        diff_1 = self.abs(offsets_v[..., 1:] - offsets_v[..., :-1]) * kernel[..., :-1]
        diff_2 = self.abs(offsets_h[:, :, 1:, :] - offsets_h[:, :, :-1, :]) * kernel[:, :, :-1, :]
        tv_loss = diff_1.sum()+diff_2.sum()
        return self.tvloss_weight * tv_loss / batch
