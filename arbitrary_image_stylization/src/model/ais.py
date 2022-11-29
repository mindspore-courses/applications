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
"""The complete real-time arbitrary image stylization model"""

from mindspore import nn

from model.style_predict import StylePrediction, style_normalization_activations
from model.transform import Transform, ConditionalStyleNorm

class Ais(nn.Cell):
    """
    The image stylize network.

    Args:
        style_prediction_bottleneck (int): Number of bottleneck channels used in style predict network. Default: 100.

    Inputs:
        - **input** (tuple) - Tuple of 2 Tensors with shape :math:`(N, 3, H_{in}, W_{in})`.
            The 2 Tensors need not to be equal in shape.

    Outputs:
        Tensor of shape :math:`(N, 3, H_{out}, W_{out})`, stylized images with values in [0, 1].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> network = Ais(args.style_prediction_bottleneck)
        >>> content = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> style = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = network((content, style))
    """
    def __init__(self, style_prediction_bottleneck=100):
        super(Ais, self).__init__()
        activation_names, activation_depths = style_normalization_activations()
        self.style_predict = StylePrediction(activation_names, activation_depths,
                                             style_prediction_bottleneck=style_prediction_bottleneck)
        self.transform = Transform(3)
        self.norm = ConditionalStyleNorm()

    def construct(self, x):
        content, style = x
        style_params, _ = self.style_predict(style)
        stylized_images = self.transform((content, self.norm, style_params))
        return stylized_images
