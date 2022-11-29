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

#include "adaptive_gridsampler_kernels.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"

using std::max;
using std::min;

namespace {
const char* GRID_SAMPLER = "GridSampler";
const uint32_t k1stInputIndex = 0;
const uint32_t k2ndInputIndex = 1;
const uint32_t k3thInputIndex = 2;
const uint32_t k4thInputIndex = 3;
const uint32_t k5thInputIndex = 4;
const uint32_t k6thInputIndex = 5;
const uint32_t kNumOfParam = 6;
const uint32_t k1stOutputIndex = 0;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
}  // namespace

namespace aicpu {
uint32_t GridSamplerKernel::Compute(CpuKernelContext &ctx) {
  uint32_t inputSize = ctx.GetInputsSize();
  if (inputSize != kNumOfParam) {
    return PARAM_INVAILD;
  }
  return GridSamplerCompute(ctx);
}

uint32_t GridSamplerKernel::GridSamplerCompute(const CpuKernelContext &ctx) {
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
  return GridSamplerComputeWithBlock(ctx, blockid, blockdim);
}

uint32_t GridSamplerKernel::GridSamplerComputeWithBlock(const CpuKernelContext &ctx,
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
  int64_t *padding_data = reinterpret_cast<int64_t *>(param[k6thInputIndex]);;

  Tensor *output = ctx.Output(k1stOutputIndex);
  float *y = reinterpret_cast<float *>(output->GetData());
  if (y == nullptr) {
    return PARAM_INVAILD;
  }

  // calculate per unit if blockdimByIndex = -1
  int64_t total = output->NumElements();
  int64_t startpos = 0;
  int64_t len = total;
  if (blockdim != 1) {
    uint32_t per_unit = std::ceil(total / blockdim);
    startpos =  blockid * per_unit;
    len = blockid < blockdim - 1 ? per_unit : (total - per_unit * (blockdim - 1));
  }

  auto output_shape = output->GetTensorShape();
  auto dim_b = output_shape->GetDimSize(0);
  auto dim_c = output_shape->GetDimSize(1);
  auto dim_h = output_shape->GetDimSize(2);
  auto dim_w = output_shape->GetDimSize(3);

  Tensor *img = ctx.Input(k1stInputIndex);
  auto img_dim_c = img->GetTensorShape()->GetDimSize(1);
  auto img_dim_h = img->GetTensorShape()->GetDimSize(2);
  auto img_dim_w = img->GetTensorShape()->GetDimSize(3);

  Tensor *kernels = ctx.Input(k2ndInputIndex);
  int k_size = sqrt(static_cast<float>(kernels->GetTensorShape()->GetDimSize(1)));
  int offset = *offset_unit_data;
  int pad = *padding_data;
  for (int i = startpos; i < startpos + len; i++) {
    auto idb = (i / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (i / (dim_h * dim_w)) % dim_c;
    auto idy = (i / dim_w) % dim_h;
    auto idx = i % dim_w;
    float result = 0;

    float w = static_cast<float>(img->GetTensorShape()->GetDimSize(3) - 2 * pad);
    float h = static_cast<float>(img->GetTensorShape()->GetDimSize(2) - 2 * pad);

    for (int k_y = 0; k_y < k_size; ++k_y) {
        for (int k_x = 0; k_x < k_size; ++k_x) {
          float offset_h = *(offsets_h_data +
                        (idb) * (dim_c * dim_h * dim_w) +
                        (k_size * k_y + k_x) * (dim_h * dim_w) +
                        (idy) * (dim_w) +
                        (idx)) * offset;

          float offset_v = *(offsets_v_data +
                        (idb) * (dim_c * dim_h * dim_w) +
                        (k_size * k_y + k_x) * (dim_h * dim_w) +
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

          float val = 0;

          float data1 = *(img_data +
                          idb * img_dim_c * img_dim_h * img_dim_w +
                          idc * img_dim_h * img_dim_w +
                          yT * img_dim_w +
                          xL);
          float data2 = *(img_data +
                          idb * img_dim_c * img_dim_h * img_dim_w +
                          idc * img_dim_h * img_dim_w +
                          yT * img_dim_w +
                          xR);
          float data3 = *(img_data +
                          idb* img_dim_c * img_dim_h * img_dim_w +
                          idc * img_dim_h * img_dim_w +
                          yB * img_dim_w +
                          xL);
          float data4 = *(img_data +
                          idb* img_dim_c * img_dim_h * img_dim_w +
                          idc * img_dim_h * img_dim_w +
                          yB * img_dim_w +
                          xR);
          val += (1 - alpha) * (1 - beta) * data1;
          val += alpha * (1 - beta) * data2;
          val += (1 - alpha) * beta * data3;
          val += alpha * beta * data4;

          result += val * (*(kernels_data +
                          (idb * dim_c * dim_h * dim_w) +
                          (k_size * k_y + k_x)*(dim_h * dim_w) +
                          (idy * dim_w) + idx));
        }
    }
    y[i] = result;
  }
  return SUCCESS;
}

REGISTER_CPU_KERNEL(GRID_SAMPLER, GridSamplerKernel);

}  // namespace aicpu
