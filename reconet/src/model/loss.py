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
"""ReCoNet loss"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from utils.reconet_utils import preprocess_for_vgg, gram_matrix, resize_optical_flow, \
    warp_optical_flow, rgb_to_luminance
from model.reconet import ReCoNet


class ReCoNetWithLoss(nn.Cell):
    """
    ReCoNet model with loss

    Args:
        model (ReCoNet): ReCoNet model
        vgg (Cell): Vgg encoder
        device_target (String): device target
        alpha (float): Content loss weight
        beta (float): Style loss weight
        gamma (float): Total variantion loss weight
        lambda_f (float): featiure temporal loss weight
        lambda_o (float): output temporal loss weight

    Returns:
        Tensor, loss of model

    Examples:
        >>> reconet = ReCoNet()
        >>> vgg_net = vgg16()
        >>> model = ReCoNetWithLoss(reconet, vgg_net, 'GPU', 1e4, 1e5, 1e-5, 1e5, 2e5)
    """

    def __init__(self,
                 model: ReCoNet,
                 vgg,
                 device_target,
                 alpha,
                 beta,
                 gamma,
                 lambda_f,
                 lambda_o):
        super(ReCoNetWithLoss, self).__init__(auto_prefix=False)
        self.backbone = model
        self.net = vgg
        self.net.set_grad(False)
        self.device = device_target
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.tv_loss = TotalVariationLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_f = lambda_f
        self.lambda_o = lambda_o

    def construct(self, frame, pre_frame, style_gram_matrices, occlusion_mask, reverse_optical_flow):
        """Calculate ReCoNet loss."""

        # ReCoNet encode and decode
        reconet_input = frame * 2 - 1
        feature_maps = self.backbone.encoder(reconet_input)
        output_frame = self.backbone.decoder(feature_maps)

        previous_input = pre_frame * 2 - 1
        previous_feature_maps = self.backbone.encoder(previous_input)
        previous_output_frame = self.backbone.decoder(previous_feature_maps)

        # Compute VGG features
        vgg_input_frame = preprocess_for_vgg(frame)
        vgg_output_frame = preprocess_for_vgg((output_frame + 1) / 2)
        input_vgg_features = self.net.encode(vgg_input_frame)
        output_vgg_features = self.net.encode(vgg_output_frame)

        vgg_previous_input_frame = preprocess_for_vgg(pre_frame)
        vgg_previous_output_frame = preprocess_for_vgg((previous_output_frame + 1) / 2)
        previous_input_vgg_features = self.net.encode(vgg_previous_input_frame)
        previous_output_vgg_features = self.net.encode(vgg_previous_output_frame)

        # compute loss
        content_loss = self.content_loss(output_vgg_features[2], input_vgg_features[2]) + \
                       self.content_loss(previous_output_vgg_features[2], previous_input_vgg_features[2])
        style_loss = self.style_loss(output_vgg_features, style_gram_matrices, self.device) + \
                     self.style_loss(previous_output_vgg_features, style_gram_matrices, self.device)
        total_var_loss = self.tv_loss(output_frame) + self.tv_loss(previous_output_frame)
        f_temp_loss = self.feature_temporal_loss(feature_maps, previous_feature_maps,
                                                 reverse_optical_flow,
                                                 occlusion_mask)
        o_temp_loss = self.output_temporal_loss(reconet_input, previous_input,
                                                output_frame, previous_output_frame,
                                                reverse_optical_flow,
                                                occlusion_mask)

        return self.alpha * content_loss + \
               self.beta * style_loss + \
               self.gamma * total_var_loss + \
               self.lambda_f * f_temp_loss + \
               self.lambda_o * o_temp_loss

    def content_loss(self, output_feature, input_feature):
        """Content loss"""
        _, c, h, w = output_feature.shape
        return self.l2_loss(output_feature, input_feature) / (c * h * w)

    def style_loss(self, vgg_feature_gram, style_gram, device_target):
        """Style loss"""
        loss = 0
        for content_fm, style_gm in zip(vgg_feature_gram, style_gram):
            loss += self.l2_loss(gram_matrix(content_fm, device_target), style_gm)
        return loss

    def feature_temporal_loss(self, feature_maps, previous_feature_maps, reverse_optical_flow, occlusion_mask):
        """Feature temporal loss"""
        _, c, h, w = feature_maps.shape

        reverse_optical_flow_resized = resize_optical_flow(reverse_optical_flow, h, w)
        occlusion_mask_resized = ops.ResizeNearestNeighbor((h, w))(occlusion_mask)
        feature_maps = occlusion_mask_resized * feature_maps
        pre_feature_maps = occlusion_mask_resized * warp_optical_flow(previous_feature_maps,
                                                                      reverse_optical_flow_resized)
        loss = self.l2_loss(feature_maps, pre_feature_maps) / (c * h * w)
        return loss

    def output_temporal_loss(self, input_frame, previous_input_frame, output_frame, previous_output_frame,
                             reverse_optical_flow, occlusion_mask):
        """Output temporal loss"""
        input_diff = input_frame - warp_optical_flow(previous_input_frame, reverse_optical_flow)
        output_diff = output_frame - warp_optical_flow(previous_output_frame, reverse_optical_flow)
        luminance_input_diff = rgb_to_luminance(input_diff)
        luminance_input_diff = ops.ExpandDims()(luminance_input_diff, 1)
        _, _, h, w = input_frame.shape
        loss = self.l2_loss(occlusion_mask * output_diff, occlusion_mask * luminance_input_diff) / (h * w)
        return loss


class TotalVariationLoss(nn.Cell):
    """
    Total variation loss

    Args:
        reduction (str): Reduction

    Returns:
        Tensor, Total variation loss
    """

    def __init__(self, reduction='sum'):
        super(TotalVariationLoss, self).__init__()
        self.reduction = reduction
        self.abs = ops.Abs()

    def construct(self, y):
        """Total variation loss"""
        return mindspore.numpy.sum(self.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
               mindspore.numpy.sum(self.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
