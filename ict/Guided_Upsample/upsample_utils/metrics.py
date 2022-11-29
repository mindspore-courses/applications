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
"""Performance metrics."""

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P


class PSNR(nn.Cell):
    """
    Calculate PSNR metrics.

    Args:
        max_val (int): The max value of input tensor.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, max_val: int):
        super(PSNR, self).__init__()
        base10 = P.Log()(mindspore.Tensor(10.0, mindspore.float32))
        max_val = P.Cast()(mindspore.Tensor(max_val), mindspore.float32)
        self.base10 = mindspore.Parameter(base10, requires_grad=False)
        self.max_val = mindspore.Parameter(20 * P.Log()(max_val) / base10, requires_grad=False)

    def __call__(self, a, b):
        a = P.Cast()(a, mindspore.float32)
        b = P.Cast()(b, mindspore.float32)
        mse = P.ReduceMean()((a - b) ** 2)
        if mse == 0:
            return mindspore.Tensor(0)
        return self.max_val - 10 * P.Log()(mse) / self.base10
