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

import os

from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp

from model.block import ReflectionPad2d

so_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "adaptive_gridsampler_cuda.so"
)

sampler_gpu_info = (CustomRegOp()
                    .input(0, "img")
                    .input(1, "kernels")
                    .input(2, "offsets_h")
                    .input(3, "offsets_v")
                    .input(4, "offset_unit")
                    .input(5, "padding")
                    .output(0, "output")
                    .dtype_format(DataType.F32_Default,
                                  DataType.F32_Default,
                                  DataType.F32_Default,
                                  DataType.F32_Default,
                                  DataType.None_Default,
                                  DataType.None_Default,
                                  DataType.F32_Default)
                    .target("GPU")
                    .get_op_info())

sampler_bprop_gpu_info = (CustomRegOp()
                          .input(0, "img")
                          .input(1, "kernels")
                          .input(2, "offsets_h")
                          .input(3, "offsets_v")
                          .input(4, "offset_unit")
                          .input(5, "padding")
                          .input(6, "grad_output")
                          .output(0, "grad_k")
                          .output(1, "grad_oh")
                          .output(2, "grad_ob")
                          .dtype_format(DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.None_Default,
                                        DataType.None_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default)
                          .target("GPU")
                          .get_op_info())


def infer_shape_backward(x1, x2, x3, x4, x5, x6, x7):
    _ = (x1, x3, x4, x5, x6, x7)
    return (x2, x2, x2)


def infer_type_backward(x1, x2, x3, x4, x5, x6, x7):
    _ = (x2, x3, x4, x5, x6, x7)
    return (x1, x1, x1)


aot_bprop = ops.Custom(
    so_path + ":adaptive_gridsampler_cuda_backward",
    infer_shape_backward,
    infer_type_backward,
    "aot",
    reg_info=sampler_bprop_gpu_info,
)


def backward(img, kernels, offsets_h, offsets_v, offset_unit, padding, out, dout):
    _ = out
    input_img = img[..., padding:-padding, padding:-padding]
    grad_k, grad_h, grad_v = aot_bprop(
        input_img, kernels, offsets_h, offsets_v, offset_unit, padding, dout
    )

    return (ops.ZerosLike()(img), grad_k, grad_h, grad_v, None, None)


def infer_shape_forward(x1, x2, x3, x4, x5, x6):
    _ = (x3, x4, x5, x6)
    shape = (x1[0], x1[1], x2[2], x2[3])
    return shape


def infer_type_forward(x1, x2, x3, x4, x5, x6):
    _ = (x2, x3, x4, x5, x6)
    return x1


class Downsampler(Cell):
    """
    Downsampler
    """
    def __init__(self, k_size):
        super().__init__()
        self.k_size = k_size
        self.ops = ops.Custom(
            so_path + ":adaptive_gridsampler_cuda_forward",
            infer_shape_forward,
            infer_type_forward,
            "aot",
            backward,
            sampler_gpu_info,
        )

    def construct(self, img, kernels, offsets_h, offsets_v, offset_unit):
        padding = self.k_size // 2
        img = ReflectionPad2d(padding)(img)

        return self.ops(img, kernels, offsets_h, offsets_v, offset_unit, padding)
