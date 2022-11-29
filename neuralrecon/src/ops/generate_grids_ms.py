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
"""Generate grids with MindSpore"""


import mindspore as ms
from mindspore import ops
from mindspore import numpy as nps


def generate_grid(n_vox, interval):
    # Create voxel grid
    grid_range = [nps.arange(0, n_vox[axis], interval) for axis in range(3)]
    grid = ops.stack(ops.meshgrid((grid_range[0], grid_range[1], grid_range[2])), indexing='ij')  # 3 dx dy dz
    grid = ops.cast(ops.expand_dims(grid, 0), ms.float32)  # 1 3 dx dy dz
    grid = grid.view(1, 3, -1)
    return ops.stop_gradient(grid)
