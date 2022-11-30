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
"""Adaptive sampler for mindspore"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import CustomRegOp, custom_info_register, DataType

from model.block import ReflectionPad2d

cust_aicpu_so_path = "cust_aicpu_kernels"

sampler_ascend_info = CustomRegOp("GridSampler") \
    .fusion_type("OPAQUE") \
    .input(0, "img")\
    .input(1, "kernels")\
    .input(2, "offsets_h")\
    .input(3, "offsets_v")\
    .input(4, "offset_unit")\
    .input(5, "padding")\
    .output(0, "output")\
    .attr("cust_aicpu", "required", "str", "cust_aicpu_kernels") \
    .dtype_format(DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.I64_Default,
                  DataType.I64_Default,
                  DataType.F32_Default) \
    .target("Ascend") \
    .get_op_info()
@custom_info_register(sampler_ascend_info)
def gridsampler():
    return

sampler_bprop_ascend_info = CustomRegOp("GridSamplerGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "img")\
    .input(1, "kernels")\
    .input(2, "offsets_h")\
    .input(3, "offsets_v")\
    .input(4, "offset_unit")\
    .input(5, "padding")\
    .input(6, "grad_output")\
    .output(0, "grad_k")\
    .output(1, "grad_oh")\
    .output(2, "grad_ob")\
    .attr("cust_aicpu", "required", "str", "cust_aicpu_kernels") \
    .dtype_format(DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.I64_Default,
                  DataType.I64_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default,
                  DataType.F32_Default) \
    .target("Ascend") \
    .get_op_info()
@custom_info_register(sampler_bprop_ascend_info)
def gridsampler_bprop():
    return

def infer_shape_forward(x1, x2, x3, x4, x5, x6, cust_attr):
    _ = (x3, x4, x5, x6, cust_attr)
    shape = (x1[0], x1[1], x2[2], x2[3])
    return shape


def infer_type_forward(x1, x2, x3, x4, x5, x6, cust_attr):
    _ = (x2, x3, x4, x5, x6, cust_attr)
    return x1

def infer_shape_backward(x1, x2, x3, x4, x5, x6, x7, cust_attr):
    _ = (x1, x3, x4, x5, x6, x7, cust_attr)
    return (x2, x2, x2)


def infer_type_backward(x1, x2, x3, x4, x5, x6, x7, cust_attr):
    _ = (x2, x3, x4, x5, x6, x7, cust_attr)
    return (x1, x1, x1)

ascend_bprop = ops.Custom(
    gridsampler_bprop,
    out_shape=infer_shape_backward,
    out_dtype=infer_type_backward,
    func_type="aicpu"
)

def backward(img, kernels, offsets_h, offsets_v, offset_unit, padding, cust_attr, out, dout):
    _ = (out, cust_attr)
    input_img = img[..., padding:-padding, padding:-padding]
    grad_k, grad_h, grad_v = ascend_bprop(
        input_img, kernels, offsets_h, offsets_v, offset_unit, padding, dout, cust_aicpu_so_path
    )

    return (ops.ZerosLike()(img), grad_k, grad_h, grad_v, 0, 0, 0)

class Downsampler(nn.Cell):
    """
    Downsampler
    """
    def __init__(self, k_size=1):
        super().__init__()
        self.k_size = k_size
        self.ops = ops.Custom(
            gridsampler,
            out_shape=infer_shape_forward,
            out_dtype=infer_type_forward,
            func_type="aicpu",
            bprop=backward
        )

    def construct(self, img, kernels, offsets_h, offsets_v, offset_unit):
        padding = self.k_size // 2
        img = ReflectionPad2d(padding)(img)
        return self.ops(img, kernels, offsets_h, offsets_v, offset_unit, padding, cust_aicpu_so_path)
