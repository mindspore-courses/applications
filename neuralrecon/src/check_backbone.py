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
"""Compare the output of backbone in PyTorch/MindSpore version"""

import time

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
import torch

from models.backbone import MnasMulti
from models.backbone_ms import MnasMulti as MnasMulti_ms

# Check results
# ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU')
ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')
model_ms = MnasMulti_ms()
state_dict_ms = load_checkpoint('./models/backbone2d_ms.ckpt')
load_param_into_net(model_ms, state_dict_ms)

model_torch = MnasMulti()
state_dict = torch.load('./models/backbone2d.ckpt')
model_torch.load_state_dict(state_dict, strict=True)
model_torch.cuda()
model_torch.eval()

arr_tmp = np.random.randint(0, 10, size=(1, 3, 640, 480)).astype(np.float32)
input_ms = ms.Tensor(arr_tmp)
input_torch = torch.Tensor(arr_tmp)

# Inference time
start_time = time.time()
for i in range(10000):
    output_ms = model_ms(input_ms)
print('inference fps:', 10000 / (time.time() - start_time))

for i in range(10000):
    output_torch = model_torch(input_torch)
print('inference fps:', 10000 / (time.time() - start_time))

# Check results
output_ms = model_ms(input_ms)
output_torch = model_torch(input_torch)

for i in range(len(output_torch)):
    total_diff = np.sum((output_torch[i].detach().cpu().numpy() - output_ms[i].asnumpy()) > 1e-4)
    print(f'Total diff for scale{i} output of backbone between torch and ms:', total_diff)
