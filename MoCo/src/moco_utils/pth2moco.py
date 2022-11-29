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
"""pytorch to MindSpore script."""

import torch
import mindspore
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


def pytorch2mindspore(path):
    """ pth file to CkPt of MindSpore"""

    par_dict = torch.load(path)['state_dict']

    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]

        print('========================py_name', name)
        name = name.replace(' net.0.weight ', ' conv1.weight ')
        name = name.replace(' net.1.running_mean ', ' bn1.moving_mean ')
        name = name.replace(' net.1.running_var ', ' bn1.moving_variance ')
        name = name.replace(' net.1.weight ', ' bn1.gamma ')
        name = name.replace(' net.1.bias ', ' bn1.beta ')
        name = name.replace(' net.3.0.conv1.weight ', ' layer1.0.conv1.weight ')
        name = name.replace(' net.3.0.bn1.running_var ', ' layer1.0.bn1.moving_mean ')
        name = name.replace(' net.3 .0.bn1.running_mean ', ' layer1.0.bn1.moving_variance ')
        name = name.replace(' net.3.0.bn1.weight ', ' layer1.0.bn1.gamma ')
        name = name.replace(' net.3.0.bn1.bias ', ' layer1.0.bn1.beta ')
        name = name.replace(' net.3.0.conv2.weight ', 'layer1.0.conv2.weight')
        name = name.replace(' net.3.0.bn2.running_mean', 'layer1.0.bn2.moving_mean')
        name = name.replace(' net.3.0.bn2.running_var', 'layer1.0.bn2.moving_variance')
        name = name.replace(' net.3.0.bn2.weight', 'layer1.0.bn2.gamma')
        name = name.replace(' net.3.1.conv1.weight', 'layer1.1.conv1.weight')
        name = name.replace(' net.3.1.bn1.running_var', 'layer1.1.bn1.moving_mean')
        name = name.replace(' net.3.1.bn1.running_mean', 'layer1.1.bn1.moving_variance')
        name = name.replace(' net.3.1.bn1.weight', 'layer1.1.bn1.gamma')
        name = name.replace(' net.3.1.bn1.bias', 'layer1.1.bn1.beta')
        name = name.replace(' net.3.1.conv2.weight', 'layer1.1.conv2.weight')
        name = name.replace(' net.3.1.bn2.running_mean', 'layer1.1.bn2.moving_mean')
        name = name.replace(' net.3.1.bn2.running_var', 'layer1.1.bn2.moving_variance')
        name = name.replace(' net.3.1.bn2.weight', 'layer1.1.bn2.gamma')
        name = name.replace(' net.3.1.bn2.bias', 'layer1.1.bn2.beta ')
        name = name.replace('net.4.0.conv1.weight', 'layer2.0.conv1.weight')
        name = name.replace('net.4.0.bn1.running_mean', 'layer2.0.bn1.moving_mean ')
        name = name.replace('net.4.0.bn1.running_var', 'layer2.0.bn1.moving_variance')
        name = name.replace('net.4.0.bn1.weight', 'layer2.0.bn1.gamma')
        name = name.replace('net.4.0.bn1.bias', 'layer2.0.bn1.beta')
        name = name.replace('net.4.0.conv2.weight', 'layer2.0.conv2.weight')
        name = name.replace('net.4.0.bn2.running_mean', 'layer2.0.bn2.moving_mean')
        name = name.replace('net.4.0.bn2.running_var', 'layer2.0.bn2.moving_variance')
        name = name.replace('net.4.0.bn2.weight', 'layer2.0.bn2.gamma')
        name = name.replace('net.4.0.bn2.bias', 'layer2.0.bn2.beta')
        name = name.replace('net.4.0.downsample.0.weight', 'layer2.0.downsample.0.weight')
        name = name.replace('net.4.0.downsample.1.running_mean', 'layer2.0.downsample.1.moving_mean')
        name = name.replace('net.4.0.downsample.1.running_var', 'layer2.0.downsample.1.moving_variance')
        name = name.replace('net.4.0.downsample.1.weight', 'layer2.0.downsample.1.gamma')
        name = name.replace('net.4.0.downsample.1.bias', 'layer2.0.downsample.1.beta')
        name = name.replace('net.4.1.conv1.weight', 'layer2.1.conv1.weight')
        name = name.replace('net.4.1.bn1.running_mean', 'layer2.1.bn1.moving_mean')
        name = name.replace('net.4.1.bn1.running_var', 'layer2.1.bn1.moving_variance')
        name = name.replace('net.4.1.bn1.weight', 'layer2.1.bn1.gamma')
        name = name.replace('net.4.1.bn1.bias', 'layer2.1.bn1.beta')
        name = name.replace('net.4.1.conv2.weight', 'layer2.1.conv2.weight')
        name = name.replace('net.4.1.bn2.running_mean', 'layer2.1.bn2.moving_mean')
        name = name.replace('net.4.1.bn2.running_var', 'layer2.1.bn2.moving_variance')
        name = name.replace('net.4.1.bn2.weight', 'layer2.1.bn2.gamma')
        name = name.replace('net.4.1.bn2.bias', 'layer2.1.bn2.beta')
        name = name.replace('net.5.0.conv1.weight', 'layer3.0.conv1.weight')
        name = name.replace('net.5.0.bn1.running_mean', 'layer3.0.bn1.moving_mean')
        name = name.replace('net.5.0.bn1.running_var', 'layer3.0.bn1.moving_variance')
        name = name.replace('net.5.0.bn1.weight', 'layer3.0.bn1.gamma')
        name = name.replace('net.5.0.bn1.bias', 'layer3.0.bn1.beta')
        name = name.replace('net.5.0.conv2.weight', 'layer3.0.conv2.weight')
        name = name.replace('net.5.0.bn2.running_mean', 'layer3.0.bn2.moving_mean')
        name = name.replace('net.5.0.bn2.running_var', 'layer3.0.bn2.moving_variance')
        name = name.replace('net.5.0.bn2.weight', 'layer3.0.bn2.gamma')
        name = name.replace('net.5.0.bn2.bias', 'layer3.0.bn2.beta')
        name = name.replace('net.5.0.downsample.0.weight', 'layer3.0.downsample.0.weight')
        name = name.replace('net.5.0.downsample.1.running_mean', 'layer3.0.downsample.1.moving_mean')
        name = name.replace('net.5.0.downsample.1.running_var', 'layer3.0.downsample.1.moving_variance')
        name = name.replace('net.5.0.downsample.1.weight', 'layer3.0.downsample.1.gamma')
        name = name.replace('net.5.0.downsample.1.bias', 'layer3.0.downsample.1.beta')
        name = name.replace('net.5.1.conv1.weight', 'layer3.1.conv1.weight')
        name = name.replace('net.5.1.bn1.running_mean', 'layer3.1.bn1.moving_mean')
        name = name.replace('net.5.1.bn1.running_var', 'layer3.1.bn1.moving_variance')
        name = name.replace('net.5.1.bn1.weight', 'layer3.1.bn1.gamma')
        name = name.replace('net.5.1.bn1.bias', 'layer3.1.bn1.beta')
        name = name.replace('net.5.1.conv2.weight', 'layer3.1.conv2.weight')
        name = name.replace('net.5.1.bn2.running_mean', 'layer3.1.bn2.moving_mean')
        name = name.replace('net.5.1.bn2.running_var', 'layer3.1.bn2.moving_variance')
        name = name.replace('net.5.1.bn2.weight', 'layer3.1.bn2.gamma')
        name = name.replace('net.5.1.bn2.bias', 'layer3.1.bn2.beta')
        name = name.replace('net.6.0.conv1.weight', 'layer4.0.conv1.weight')
        name = name.replace('net.6.0.bn1.running_mean', 'layer4.0.bn1.moving_mean')
        name = name.replace('net.6.0.bn1.running_var', 'layer4.0.bn1.moving_variance')
        name = name.replace('net.6.0.bn1.weight', 'layer4.0.bn1.gamma')
        name = name.replace('net.6.0.bn1.bias', 'layer4.0.bn1.beta')
        name = name.replace('net.6.0.conv2.weight', 'layer4.0.conv2.weight')
        name = name.replace('net.6.0.bn2.running_mean', 'layer4.0.bn2.moving_mean')
        name = name.replace('net.6.0.bn2.running_var', 'layer4.0.bn2.moving_variance')
        name = name.replace('net.6.0.bn2.weight', 'layer4.0.bn2.gamma')
        name = name.replace('net.6.0.bn2.bias', 'layer4.0.bn2.beta')
        name = name.replace('net.6.0.downsample.0.weight', 'layer4.0.downsample.0.weight')
        name = name.replace('net.6.0.downsample.1.running_mean', 'layer4.0.downsample.1.moving_mean')
        name = name.replace('net.6.0.downsample.1.running_var', 'layer4.0.downsample.1.moving_variance')
        name = name.replace('net.6.0.downsample.1.weight', 'layer4.0.downsample.1.gamma')
        name = name.replace('net.6.0.downsample.1.bias', 'layer4.0.downsample.1.beta')
        name = name.replace('net.6.1.conv1.weight', 'layer4.1.conv1.weight')
        name = name.replace('net.6.1.bn1.running_mean', 'layer4.1.bn1.moving_mean')
        name = name.replace('net.6.1.bn1.running_var', 'layer4.1.bn1.moving_variance')
        name = name.replace('net.6.1.bn1.weight', 'layer4.1.bn1.gamma')
        name = name.replace('net.6.1.bn1.bias', 'layer4.1.bn1.beta')
        name = name.replace('net.6.1.conv2.weight', 'layer4.1.conv2.weight')
        name = name.replace('net.6.1.bn2.running_mean', 'layer4.1.bn2.moving_mean')
        name = name.replace('net.6.1.bn2.running_var', 'layer4.1.bn2.moving_variance')
        name = name.replace('net.6.1.bn2.weight', 'layer4.1.bn2.gamma')
        name = name.replace('net.6.1.bn2.bias', 'layer4.1.bn2.beta')
        name = name.replace('net.9.weight', 'fc.weight')
        name = name.replace('net.9.bias', 'fc.bias')

        name = name.replace("net.", "layer")
        print('========================ms_name', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.cpu().numpy(), mindspore.float32)
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'pth2ckpt.ckpt')

path_pth = 'model_last.pth'
pytorch2mindspore(path_pth)
