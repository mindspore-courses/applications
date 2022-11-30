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

#include <string>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

#include "adaptive_gridsampler_grad_kernels.h"

using std::max;
using std::min;

namespace {
const char* GRID_SAMPLER_GRAD = "GridSamplerGrad";
const uint32_t k1stInputIndex = 0;
const uint32_t k2ndInputIndex = 1;
const uint32_t k3thInputIndex = 2;
const uint32_t k4thInputIndex = 3;
const uint32_t k5thInputIndex = 4;
const uint32_t k6thInputIndex = 5;
const uint32_t k7thInputIndex = 6;
const uint32_t kNumOfParam = 7;
const uint32_t k1stOutputIndex = 0;
const uint32_t k2ndOutputIndex = 1;
const uint32_t k3thOutputIndex = 2;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
}  // namespace

namespace aicpu {
uint32_t GridSamplerGradKernel::Compute(CpuKernelContext &ctx) {
  uint32_t inputSize = ctx.GetInputsSize();
  if (inputSize != kNumOfParam) {
    return PARAM_INVAILD;
  }
  return GridSamplerGradCompute(ctx);
}

uint32_t GridSamplerGradKernel::GridSamplerGradCompute(const CpuKernelContext &ctx) {
  // Get blockid and blockdim
  uint32_t blockid;
  uint32_t blockdim;
  AttrValue *block_id_ptr = ctx.GetAttr("block_id");
  AttrValue *block_dim_ptr = ctx.GetAttr("block_num");
  // check block_id and block_num
  if (block_id_ptr == nullptr || block_dim_ptr == nullptr) {
    blockid = 0;
    blockdim = 1;
  } else {
    blockid = block_id_ptr->GetInt();
    blockdim = block_dim_ptr->GetInt();
  }
  if (blockid >= blockdim || blockid < 0) {
    blockid = 0;
    blockdim = 1;
  }
  return GridSamplerGradComputeWithBlock(ctx, blockid, blockdim);
}

uint32_t GridSamplerGradKernel::GridSamplerGradComputeWithBlock(const CpuKernelContext &ctx,
                                                uint32_t blockid, uint32_t blockdim) {
  std::vector<void *> param;
  for (int i = 0; i < kNumOfParam; i++) {
    Tensor * tensor = ctx.Input(i);
    if (tensor == nullptr) {
      return PARAM_INVAILD;
    }
    void *data = tensor->GetData();
    if (data == nullptr) {
      return PARAM_INVAILD;
    }
    param.emplace_back(data);
  }

  float *img_data = reinterpret_cast<float *>(param[k1stInputIndex]);
  float *kernels_data = reinterpret_cast<float *>(param[k2ndInputIndex]);
  float *offsets_h_data = reinterpret_cast<float *>(param[k3thInputIndex]);
  float *offsets_v_data = reinterpret_cast<float *>(param[k4thInputIndex]);
  int64_t *offset_unit_data = reinterpret_cast<int64_t *>(param[k5thInputIndex]);
  int64_t *padding_data = reinterpret_cast<int64_t *>(param[k6thInputIndex]);

  float *dout_data = reinterpret_cast<float *>(param[k7thInputIndex]);

  Tensor *kernels_grad = ctx.Output(k1stOutputIndex);
  float *kernels_grad_data = reinterpret_cast<float *>(kernels_grad->GetData());
  if (kernels_grad_data == nullptr) {
    return PARAM_INVAILD;
  }
  Tensor *offsets_h_grad = ctx.Output(k2ndOutputIndex);
  float *offsets_h_grad_data = reinterpret_cast<float *>(offsets_h_grad->GetData());
  if (offsets_h_grad_data == nullptr) {
    return PARAM_INVAILD;
  }
  Tensor *offsets_v_grad = ctx.Output(k3thOutputIndex);
  float *offsets_v_grad_data = reinterpret_cast<float *>(offsets_v_grad->GetData());
  if (offsets_v_grad_data == nullptr) {
    return PARAM_INVAILD;
  }
  auto dim_b = kernels_grad->GetTensorShape()->GetDimSize(0);
  auto dim_c = kernels_grad->GetTensorShape()->GetDimSize(1);
  auto dim_h = kernels_grad->GetTensorShape()->GetDimSize(2);
  auto dim_w = kernels_grad->GetTensorShape()->GetDimSize(3);

  Tensor *img = ctx.Input(k1stInputIndex);
  auto img_dim_c = img->GetTensorShape()->GetDimSize(1);
  auto img_dim_h = img->GetTensorShape()->GetDimSize(2);
  auto img_dim_w = img->GetTensorShape()->GetDimSize(3);

  Tensor *dout = ctx.Input(k7thInputIndex);
  auto dout_dim_c = dout->GetTensorShape()->GetDimSize(1);
  auto dout_dim_h = dout->GetTensorShape()->GetDimSize(2);
  auto dout_dim_w = dout->GetTensorShape()->GetDimSize(3);

  int k_size = sqrt(static_cast<float>(dim_c));
  int offset = *offset_unit_data;
  int pad = *padding_data;

  float w = static_cast<float>(img_dim_w - 2 * pad);
  float h = static_cast<float>(img_dim_h - 2 * pad);

  int64_t total = kernels_grad->NumElements();
  int64_t startpos = 0;
  int64_t len = total;
  if (blockdim != 1) {
    uint32_t per_unit = std::ceil(total / blockdim);
    startpos =  blockid * per_unit;
    len = blockid < blockdim - 1 ? per_unit : (total - per_unit * (blockdim - 1));
  }
  for (int i = startpos; i < startpos + len; i++) {
    auto idb = (i / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (i / (dim_h * dim_w)) % dim_c;
    auto idy = (i / dim_w) % dim_h;
    auto idx = i % dim_w;
    int k_y = idc / k_size;
    int k_x = idc % k_size;

    float offset_h = *(offsets_h_data +
                      (idb) * (dim_c * dim_h * dim_w) +
                      (idc) * (dim_h * dim_w) +
                      (idy) * (dim_w) +
                      (idx)) * offset;

    float offset_v = *(offsets_v_data +
                      (idb) * (dim_c * dim_h * dim_w) +
                      (idc) * (dim_h * dim_w) +
                      (idy) * (dim_w) +
                      (idx)) * offset;

    float p_x = static_cast<float>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
    float p_y = static_cast<float>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
    float alpha = p_x - floor(p_x);
    float beta = p_y - floor(p_y);
    int xL = max(min(static_cast<int>(floor(p_x)), static_cast<int>(w + 2 * pad - 1)), 0);
    int xR = max(min(xL + 1, static_cast<int>(w + 2 * pad - 1)), 0);
    int yT = max(min(static_cast<int>(floor(p_y)), static_cast<int>(h + 2 * pad - 1)), 0);
    int yB = max(min(yT + 1, static_cast<int>(h + 2 * pad - 1)), 0);

    float grad_kernels = 0;
    float grad_offset_h = 0;
    float grad_offset_v = 0;
    float c_br = 0;
    for (int c = 0; c < img_dim_c; ++c) {
        float c_tl = *(img_data +
                      (idb) * (img_dim_c * img_dim_h * img_dim_w) +
                      (c) * (img_dim_h * img_dim_w) +
                      (yT) * (img_dim_w) +
                      (xL));
        float c_tr = *(img_data +
                      (idb) * (img_dim_c * img_dim_h * img_dim_w) +
                      (c) * (img_dim_h * img_dim_w) +
                      (yT) * (img_dim_w) +
                      (xR));
        float c_bl = *(img_data +
                      (idb) * (img_dim_c * img_dim_h * img_dim_w) +
                      (c) * (img_dim_h * img_dim_w) +
                      (yB) * (img_dim_w) +
                      (xL));
        float c_br = *(img_data +
                      (idb) * (img_dim_c * img_dim_h * img_dim_w) +
                      (c) * (img_dim_h * img_dim_w) +
                      (yB) * (img_dim_w) +
                      (xR));

        float grad = 0;
        float kernel = *(kernels_data +
                        (idb) * (dim_c * dim_h * dim_w) +
                        (idc) * (dim_h * dim_w) +
                        (idy) * (dim_w) +
                        (idx));
        float dout_elem = *(dout_data + idb * dout_dim_c * dout_dim_h * dout_dim_w +
                           (c * dout_dim_h * dout_dim_w) + idy * dout_dim_w + idx);

        grad += (1 - alpha) * (1 - beta) * c_tl;
        grad += alpha * (1 - beta) * c_tr;
        grad += (1 - alpha) * beta * c_bl;
        grad += alpha * beta * c_br;
        grad_kernels += grad * dout_elem;

        grad = (beta - 1) * c_tl + (1 - beta) * c_tr - beta * c_bl + beta * c_br;
        grad_offset_h += kernel * grad * dout_elem * offset;

        grad = (alpha - 1) * c_tl - alpha * c_tr + (1 - alpha) * c_bl + alpha * c_br;
        grad_offset_v += kernel * grad * dout_elem * offset;
    }
    kernels_grad_data[i] = grad_kernels;
    offsets_h_grad_data[i] = grad_offset_h;
    offsets_v_grad_data[i] = grad_offset_v;
  }
  return SUCCESS;
}

REGISTER_CPU_KERNEL(GRID_SAMPLER_GRAD, GridSamplerGradKernel);

}  // namespace aicpu
