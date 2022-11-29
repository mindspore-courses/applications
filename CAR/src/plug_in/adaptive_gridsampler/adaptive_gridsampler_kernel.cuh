/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ADAPTIVE_GRIDSAMPLER_KERNEL_CUH
#define ADAPTIVE_GRIDSAMPLER_KERNEL_CUH

#include <ATen/ATen.h>

void adaptive_gridsampler_kernel_forward(
    const at::Tensor &img,
    const at::Tensor &kernels,
    const at::Tensor &offsets_h,
    const at::Tensor &offsets_v,
    const int *offset_unit,
    const int *padding,
    at::Tensor *output);

void adaptive_gridsampler_kernel_backward(
    const at::Tensor &img,
    const at::Tensor &kernels,
    const at::Tensor &offsets_h,
    const at::Tensor &offsets_v,
    const int *offset_unit,
    const int *padding,
    const at::Tensor &gradOutput,
    at::Tensor *gradInput_kernels,
    at::Tensor *gradInput_offsets_h,
    at::Tensor *gradInput_offsets_v);

#endif
