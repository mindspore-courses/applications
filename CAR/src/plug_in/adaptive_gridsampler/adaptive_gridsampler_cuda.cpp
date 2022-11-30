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

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <string.h>
#include <cuda_runtime.h>

#include "adaptive_gridsampler_kernel.cuh"


int8_t GetDtype(const std::string &dtypes) {
    int8_t type = 6;
    std::unordered_map<std::string, int8_t> m {
        {"uint8", 0}, {"int8", 1},
        {"int16", 2}, {"int32", 3},
        {"int64", 4}, {"float16", 5},
        {"float32", 6}, {"float64", 7}};
    if (m.count(dtypes)) {
        type = m[dtypes];
    }
    return type;
}

at::Tensor get_one_torch_tensors(void *params, int ndims, int64_t *shapes, const char *dtypes, c10::Device device) {
    std::vector<int64_t> size;
    at::Tensor tensor;
    for (int i = 0; i < ndims; i++) {
        size.push_back(shapes[i]);
    }
    int8_t type = GetDtype(dtypes);
    auto option = at::TensorOptions().dtype(static_cast<c10::ScalarType>(type)).device(device);
    tensor = at::from_blob(params, size, option);
    return tensor;
}

extern "C" int adaptive_gridsampler_cuda_forward(
    int nparam,
    void **params,
    int *ndims,
    int64_t **shapes,
    const char **dtypes,
    void *stream,
    void *extra) {

    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(custream);

    auto image = get_one_torch_tensors(params[0], ndims[0], shapes[0], dtypes[0], c10::kCUDA);
    auto kernels = get_one_torch_tensors(params[1], ndims[1], shapes[1], dtypes[1], c10::kCUDA);
    auto offsets_h = get_one_torch_tensors(params[2], ndims[2], shapes[2], dtypes[2], c10::kCUDA);
    auto offsets_v = get_one_torch_tensors(params[3], ndims[3], shapes[3], dtypes[3], c10::kCUDA);
    auto offset_unit = static_cast<int *>(params[4]);
    auto padding = static_cast<int *>(params[5]);
    auto at_output = get_one_torch_tensors(params[6], ndims[6], shapes[6], dtypes[6], c10::kCUDA);

    adaptive_gridsampler_kernel_forward(image, kernels, offsets_h, offsets_v, offset_unit, padding, &at_output);

    cudaDeviceSynchronize();
    return 0;
}

extern "C" int adaptive_gridsampler_cuda_backward(
    int nparam,
    void **params,
    int *ndims,
    int64_t **shapes,
    const char **dtypes,
    void *stream,
    void *extra) {

    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(custream);

    auto image = get_one_torch_tensors(params[0], ndims[0], shapes[0], dtypes[0], c10::kCUDA);
    auto kernels = get_one_torch_tensors(params[1], ndims[1], shapes[1], dtypes[1], c10::kCUDA);
    auto offsets_h = get_one_torch_tensors(params[2], ndims[2], shapes[2], dtypes[2], c10::kCUDA);
    auto offsets_v = get_one_torch_tensors(params[3], ndims[3], shapes[3], dtypes[3], c10::kCUDA);
    auto offset_unit = static_cast<int *>(params[4]);
    auto padding = static_cast<int *>(params[5]);
    auto grad_output = get_one_torch_tensors(params[6], ndims[6], shapes[6], dtypes[6], c10::kCUDA);

    auto grad_kernels = get_one_torch_tensors(params[7], ndims[7], shapes[7], dtypes[7], c10::kCUDA);
    auto grad_oh = get_one_torch_tensors(params[8], ndims[8], shapes[8], dtypes[8], c10::kCUDA);
    auto grad_ov = get_one_torch_tensors(params[9], ndims[9], shapes[9], dtypes[9], c10::kCUDA);

    adaptive_gridsampler_kernel_backward(
        image,
        kernels,
        offsets_h,
        offsets_v,
        offset_unit,
        padding,
        grad_output,
        &grad_kernels,
        &grad_oh,
        &grad_ov);

    cudaDeviceSynchronize();

    return 0;
}
