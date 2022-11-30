"""implicit reconstruct utils"""
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
import mindspore as ms
from mindspore import ops, nn, Parameter


def create_grid_3d(min_num, max_num, steps):
    """create 3d grid"""
    if isinstance(min_num) is int:
        min_num = (min_num, min_num, min_num)  # (x, y, z)
    if isinstance(max_num) is int:
        max_num = (max_num, max_num, max_num)  # (x, y)
    if isinstance(steps) is int:
        steps = (steps, steps, steps)  # (x, y, z)
    arrange_x = ops.linspace(min_num[0], max_num[0], steps[0]).astype(ms.int64)
    arrange_y = ops.linspace(min_num[1], max_num[1], steps[1]).astype(ms.int64)
    arrange_z = ops.linspace(min_num[2], max_num[2], steps[2]).astype(ms.int64)
    grid_d, gird_h, grid_w = ops.meshgrid((arrange_z, arrange_y, arrange_x))
    coords = ops.stack([grid_w, gird_h, grid_d])  # [2, steps[0], steps[1], steps[2]]
    coords = coords.view(3, -1).t()  # [N, 3]
    return coords


class SmoothConv3D(nn.Cell):
    """smmoth conv 3d"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert (
            kernel_size % 2 == 1
        ), "kernel_size for smooth_conv must be odd: {3, 5, ...}"
        self.padding = (kernel_size - 1) // 2

        weight = ms.ops.ones(
            (in_channels, out_channels, kernel_size, kernel_size, kernel_size),
            dtype=ms.float32,
        ) / (kernel_size ** 3)
        self.weight = Parameter(weight, requires_grad=False)

    def forward(self, input_data):
        """forward"""
        return nn.Conv3d(input_data, self.weight, padding=self.padding, pad_mode="pad")
