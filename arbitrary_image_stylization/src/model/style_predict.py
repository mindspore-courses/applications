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
""" Style predict network."""

from mindspore import nn, ops

from model.inception_v3 import InceptionV3

def style_normalization_activations(pre_name='transformer', post_name='StyleNorm', alpha=1.0):
    """Get names and depths of each layer."""
    scope_names = [
        'residual/residual1/conv1', 'residual/residual1/conv2',
        'residual/residual2/conv1', 'residual/residual2/conv2',
        'residual/residual3/conv1', 'residual/residual3/conv2',
        'residual/residual4/conv1', 'residual/residual4/conv2',
        'residual/residual5/conv1', 'residual/residual5/conv2',
        'expand/conv1/conv', 'expand/conv2/conv', 'expand/conv3/conv'
    ]
    scope_names = ['{}/{}/{}'.format(pre_name, name, post_name) for name in scope_names]
    # 10 convolution layers of 'residual/residual*/conv*' have the same depth.
    depths = [int(alpha * 128)] * 10 + [int(alpha * 64), int(alpha * 32), 3]
    return scope_names, depths

class StylePrediction(nn.Cell):
    """
    Map style images to the style embedding (beta and gamma parameters).

    Args:
        activation_names (list[string]): Scope names of the  corresponding activations of the transformer network.
        activation_depths (list[int]): Shapes of the corresponding activations of the transformer network.
        style_prediction_bottlenneck (int): Output channels of bottleneck layer.

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, 3, H_{in}, W_{in})`.

    Outputs:
        Tuple of 1 dict and 1 Tensor.
        - **style_params** (dict) - dict mapping activations names and the corresponding parameters.
        - **x** (Tensor) - Tensor of shape :math:`(N, style_prediction_bottleneck, H_{out}, W_{out})`

    Example:
        >>> activation_names = ['residual/residual1/conv1', 'residual/residual1/conv2']
        >>> activation_depths = [128, 64]
        >>> style_predict = StylePrediction(activation_names, activation_depths)
    """
    def __init__(self, activation_names, activation_depths, style_prediction_bottleneck=100):
        super(StylePrediction, self).__init__()
        self.encoder = InceptionV3(in_channels=3)
        self.bottleneck = nn.SequentialCell([nn.Conv2d(768, style_prediction_bottleneck, kernel_size=1, has_bias=True)])
        self.activation_depths = activation_depths
        self.activation_names = activation_names
        self.beta = nn.CellList()
        self.gamma = nn.CellList()
        self.squeeze = ops.Squeeze((2, 3))
        for i in activation_depths:
            self.beta.append(nn.Conv2d(style_prediction_bottleneck, i, kernel_size=1, has_bias=True))
            self.gamma.append(nn.Conv2d(style_prediction_bottleneck, i, kernel_size=1, has_bias=True))

    def construct(self, x):
        """ Forward process """
        reduce_mean = ops.ReduceMean(keep_dims=True)
        x = self.encoder(x)
        x = reduce_mean(x, (2, 3))
        x = self.bottleneck(x)
        style_params = {}
        for i in range(len(self.activation_depths)):
            beta = self.beta[i](x)
            beta = self.squeeze(beta)
            style_params[self.activation_names[i] + '/beta'] = beta
            gamma = self.gamma[i](x)
            gamma = self.squeeze(gamma)
            style_params[self.activation_names[i] + '/gamma'] = gamma
        return style_params, x
