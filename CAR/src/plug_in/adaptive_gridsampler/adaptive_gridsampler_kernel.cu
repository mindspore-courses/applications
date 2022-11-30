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

#include <stdio.h>
#include <ATen/ATen.h>
#include <helper_cuda.h>
#include <algorithm>

#include "adaptive_gridsampler_kernel.cuh"
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void kernel_adaptive_gridsampler_update_output(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> img,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> kernels,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> offsets_h,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> offsets_v,
    const int *ptr_offset_unit,
    const int *ptr_padding,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> output,
    const size_t n) {
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_idx >= n)
        return;

    auto dim_b = output.size(0);
    auto dim_c = output.size(1);
    auto dim_h = output.size(2);
    auto dim_w = output.size(3);

    auto idb = (global_idx / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (global_idx / (dim_h * dim_w)) % dim_c;
    auto idy = (global_idx / dim_w) % dim_h;
    auto idx = global_idx % dim_w;

    if (idx >= dim_w || idy >= dim_h)
        return;

    int offset_unit = *ptr_offset_unit;
    int padding = *ptr_padding;

    int k_size = sqrt(static_cast<float>(kernels.size(1)));
    float w = static_cast<float>(img.size(3) - 2 * padding);
    float h = static_cast<float>(img.size(2) - 2 * padding);

    scalar_t result = 0;
    for (int k_y = 0; k_y < k_size; ++k_y) {
        for (int k_x = 0; k_x < k_size; ++k_x) {
            scalar_t offset_h = offsets_h[idb][k_size * k_y + k_x][idy][idx] * offset_unit;
            scalar_t offset_v = offsets_v[idb][k_size * k_y + k_x][idy][idx] * offset_unit;

            scalar_t p_x = static_cast<scalar_t>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
            scalar_t p_y = static_cast<scalar_t>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
            scalar_t alpha = p_x - floor(p_x);
            scalar_t beta = p_y - floor(p_y);

            int xL = max(min(static_cast<int>(floor(p_x)), static_cast<int>(w + 2 * padding - 1)), 0);
            int xR = max(min(xL + 1, static_cast<int>(w + 2 * padding - 1)), 0);
            int yT = max(min(static_cast<int>(floor(p_y)), static_cast<int>(h + 2 * padding - 1)), 0);
            int yB = max(min(yT + 1, static_cast<int>(h + 2 * padding - 1)), 0);

            scalar_t val = 0;
            val += (1 - alpha) * (1 - beta) * img[idb][idc][yT][xL];
            val += alpha * (1 - beta) * img[idb][idc][yT][xR];
            val += (1 - alpha) * beta * img[idb][idc][yB][xL];
            val += alpha * beta * img[idb][idc][yB][xR];

            result += val * kernels[idb][k_size * k_y + k_x][idy][idx];
        }
    }
    output[idb][idc][idy][idx] = result;
}

void adaptive_gridsampler_kernel_forward(
    const at::Tensor &img,
    const at::Tensor &kernels,
    const at::Tensor &offsets_h,
    const at::Tensor &offsets_v,
    const int *offset_unit,
    const int *padding,
    at::Tensor *output) {

    kernel_adaptive_gridsampler_update_output<float><<<((*output).numel() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        img.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        kernels.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        offsets_h.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        offsets_v.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        offset_unit,
        padding,
        (*output).packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        (*output).numel());

    checkCudaErrors(cudaGetLastError());
}

template <typename scalar_t>
__global__ void kernel_adaptive_gridsampler_backward(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> img,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> kernels,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> offsets_h,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> offsets_v,
    const int *ptr_offset_unit,
    const int *ptr_padding,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> gradOutput,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> gradInput_kernels,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> gradInput_offsets_h,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> gradInput_offsets_v,
    const size_t n) {
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_idx >= n)
        return;

    auto dim_b = gradInput_kernels.size(0);
    auto dim_c = gradInput_kernels.size(1);
    auto dim_h = gradInput_kernels.size(2);
    auto dim_w = gradInput_kernels.size(3);

    auto idb = (global_idx / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (global_idx / (dim_h * dim_w)) % dim_c;
    auto idy = (global_idx / dim_w) % dim_h;
    auto idx = global_idx % dim_w;

    if (idx >= dim_w || idx >= dim_h)
        return;

    int k_size = sqrt(static_cast<float>(dim_c));
    int k_y = idc / k_size;
    int k_x = idc % k_size;

    int offset_unit = *ptr_offset_unit;
    int padding = *ptr_padding;

    scalar_t offset_h = offsets_h[idb][idc][idy][idx] * offset_unit;
    scalar_t offset_v = offsets_v[idb][idc][idy][idx] * offset_unit;

    float w = static_cast<float>(img.size(3) - 2 * padding);
    float h = static_cast<float>(img.size(2) - 2 * padding);

    scalar_t p_x = static_cast<scalar_t>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
    scalar_t p_y = static_cast<scalar_t>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
    scalar_t alpha = p_x - floor(p_x);
    scalar_t beta = p_y - floor(p_y);

    int xL = max(min(static_cast<int>(floor(p_x)), static_cast<int>(w + 2 * padding - 1)), 0);
    int xR = max(min(xL + 1, static_cast<int>(w + 2 * padding - 1)), 0);
    int yT = max(min(static_cast<int>(floor(p_y)), static_cast<int>(h + 2 * padding - 1)), 0);
    int yB = max(min(yT + 1, static_cast<int>(h + 2 * padding - 1)), 0);

    scalar_t grad_kernels = 0;
    scalar_t grad_offset_h = 0;
    scalar_t grad_offset_v = 0;
    for (int c = 0; c < img.size(1); ++c) {
        scalar_t c_tl = img[idb][c][yT][xL];
        scalar_t c_tr = img[idb][c][yT][xR];
        scalar_t c_bl = img[idb][c][yB][xL];
        scalar_t c_br = img[idb][c][yB][xR];

        scalar_t grad = 0;
        grad += (1 - alpha) * (1 - beta) * c_tl;
        grad += alpha * (1 - beta) * c_tr;
        grad += (1 - alpha) * beta * c_bl;
        grad += alpha * beta * c_br;
        grad_kernels += grad * gradOutput[idb][c][idy][idx];

        grad = (beta - 1) * c_tl + (1 - beta) * c_tr - beta * c_bl + beta * c_br;
        grad_offset_h += kernels[idb][idc][idy][idx] * grad * gradOutput[idb][c][idy][idx] * offset_unit;

        grad = (alpha - 1) * c_tl - alpha * c_tr + (1 - alpha) * c_bl + alpha * c_br;
        grad_offset_v += kernels[idb][idc][idy][idx] * grad * gradOutput[idb][c][idy][idx] * offset_unit;
    }

    gradInput_kernels[idb][idc][idy][idx] = grad_kernels;

    gradInput_offsets_h[idb][idc][idy][idx] = grad_offset_h;
    gradInput_offsets_v[idb][idc][idy][idx] = grad_offset_v;
}

void adaptive_gridsampler_kernel_backward(
    const at::Tensor &img,
    const at::Tensor &kernels,
    const at::Tensor &offsets_h,
    const at::Tensor &offsets_v,
    const int *offset_unit,
    const int *padding,
    const at::Tensor &gradOutput,
    at::Tensor *grad_k,
    at::Tensor *gradInput_offsets_h,
    at::Tensor *gradInput_offsets_v) {

    kernel_adaptive_gridsampler_backward<float><<<((*grad_k).numel() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0>>>(
        img.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        kernels.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        offsets_h.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        offsets_v.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        offset_unit,
        padding,
        gradOutput.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        (*grad_k).packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        (*gradInput_offsets_h).packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        (*gradInput_offsets_v).packed_accessor32<float, 4, at::RestrictPtrTraits>(),
        (*grad_k).numel());

    checkCudaErrors(cudaGetLastError());
}
