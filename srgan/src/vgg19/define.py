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

"""Structure of VGG19"""

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net

__all__ = ["vgg19"]

class VGG(nn.Cell):
    """
    Structure of VGG19 network.

    Args:
        features (nn.SequentialCell([*layers])):  The layers of VGG19 network.
    """
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    def construct(self, x):
        x = self.features(x)
        return x

def make_layers(mycfg, batch_norm=False):
    """
    Generate all layers of VGG19 network.
    You can download the pre-trained model in the following link.
    https://download.mindspore.cn/model_zoo/converted_pretrained/vgg/vgg19-0-97_5004.ckpt

    Args:
        mycfg (list): The structure of VGG19 network.
        batch_norm (bool): Whether to use BatchNormalization. Default: False.

    Returns:
        nn.SequentialCell([*layers]) (nn.SequentialCell): The layers of VGG19 network.
    """
    layers = []
    in_channels = 3
    for v in mycfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3,
                               pad_mode='pad', padding=1, has_bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(num_features=v, momentum=0.9), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell([*layers])

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(vgg_ckpt):
    """
    VGG 19-layer model (configuration "19"),  pre-trained on ImageNet.

    Args:
        vgg_ckpt (str): The path of vgg19 checkpoint file.

    Returns:
        model (VGG): VGG19 network with pre-trained weights.
    """
    model = VGG(make_layers(cfg['19']))
    params = load_checkpoint(vgg_ckpt)

    # modify the name of pre-trained model's layer
    modify_num = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
    for i in modify_num:
        bias_old = f'network.backbone.layers.{i}.bias'
        bias_new = f'features.{i}.bias'
        weight_old = f'network.backbone.layers.{i}.weight'
        weight_new = f'features.{i}.weight'
        params[weight_new] = params.pop(weight_old)
        params[bias_new] = params.pop(bias_old)

    load_param_into_net(model, params)
    return model
