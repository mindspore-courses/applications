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
""" Loss methods for real-time arbitrary image stylization model."""

import mindspore
from mindspore import ops, nn

from model.vgg import VGG

class TotalLoss(nn.Cell):
    r"""
    Compute the total loss, which is composed of content loss and style loss.
    Content loss can be computed by:
    .. math::
        L_c(x, c) = \sum_{j \in C} \frac 1 {n_j} \left\| f_j(x) - f_j(c) \right\|_2^2
    Style loss can be computed by:
    .. math::
        L_s(x, s) = \sum_{i \in S} \frac 1 {n_i} \left\| G \left[ f_i(x) \right] - G \left[ f_i(s) \right] \right\|_F^2
    where C and S denote lower and higher layer in VGG model, which was defined in content_weights and style_weights.
    G denotes Gram matrix, which is a square, symmetric matrix measuring the spatially averaged correlation structure
    across the filters within a layer's activations.
    Args:
        in_channel (int): number of input channels.
        content_weights (dict): a dict mapping layer names to their associated content loss weight.
        style_weights (dict): a dict mapping layer names to their associated  style loss weight.

    Inputs:
        - **content** (Tensor) - Tensor of shape :math:`(N, 3, H_{in_c}, W_{in_c})`.
        - **style** (Tensor) - Tensor of shape :math:`(N, 3, H_{in_s}, W_{in_s})`.
        - **stylized** (Tensor) - Tensor of shape :math:`(N, 3, H_{in_o}, W_{in_o})`.

    Outputs:
        Tensor of a single value.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> content_weight = {"vgg_16/conv3": 1}
        >>> style_weight = {"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3}
        >>> loss = TotalLoss(3, content_weight, style_weight)
    """
    def __init__(self, in_channel, content_weights, style_weights):
        super(TotalLoss, self).__init__()
        self.encoder = VGG(in_channel)
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.matmul = ops.BatchMatMul(transpose_b=True)
        self.l2_loss = nn.MSELoss()

    def content_loss(self, content_end_points, stylized_end_points, content_weights):
        """Get content distance in representational space."""
        total_content_loss = 0
        content_loss_dict = {}
        reduce_mean = ops.ReduceMean()
        for name, weights in content_weights.items():
            loss = reduce_mean((content_end_points[name] - stylized_end_points[name]) ** 2)
            weighted_loss = weights * loss
            content_loss_dict['content_loss/' + name] = loss
            content_loss_dict['weighted_content_loss/' + name] = weighted_loss
            total_content_loss += weighted_loss
        content_loss_dict['total_content_loss'] = total_content_loss

        return total_content_loss, content_loss_dict

    def style_loss(self, style_end_points, stylized_end_points, style_weights):
        """Get style distance in representational space."""
        reduce_mean = ops.ReduceMean()
        total_style_loss = 0
        style_loss_dict = {}
        for name, weights in style_weights.items():
            loss = reduce_mean(
                (self.get_matrix(stylized_end_points[name]) - self.get_matrix(style_end_points[name])) ** 2
            )
            weighted_loss = weights * loss
            style_loss_dict['style_loss/' + name] = loss
            style_loss_dict['weighted_style_loss/' + name] = weighted_loss
            total_style_loss += weighted_loss
        style_loss_dict['total_style_loss'] = total_style_loss
        return total_style_loss, style_loss_dict

    def get_matrix(self, feature):
        """Computes the Gram matrix for a set of feature maps."""
        batch_size, channels, height, width = feature.shape
        denominator = float(height * width)
        fill = ops.Fill()
        denominator = fill(mindspore.float32, (batch_size, channels, channels), denominator)
        feature_map = feature.reshape((batch_size, channels, height * width))
        matrix = self.matmul(feature_map.astype("float16"), feature_map.astype("float16"))
        div = ops.Div()
        return div(matrix, denominator)

    def construct(self, content, style, stylized):
        content_end_points = self.encoder(content)
        style_end_points = self.encoder(style)
        stylized_end_points = self.encoder(stylized)
        total_content_loss, _ = self.content_loss(content_end_points, stylized_end_points, self.content_weights)
        total_style_loss, _ = self.style_loss(style_end_points, stylized_end_points, self.style_weights)
        loss = total_content_loss + total_style_loss
        return loss
